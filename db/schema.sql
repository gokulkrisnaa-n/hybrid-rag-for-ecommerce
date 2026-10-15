-- =============================================================================
-- Kith hybrid RAG: database schema
--
-- One ParadeDB (Postgres) database, two schemas:
--   raw   : what the fetcher stores, untouched (Shopify JSON + run bookkeeping)
--   clean : what the parser produces (validated CleanProduct rows) + embeddings
--
-- Pipeline, each step reading what the previous one left here:
--   fetch -> raw.products   (upsert by product_id; rewrite only if payload_hash changed)
--   parse -> clean.*        (rows where raw.products.parsed_hash <> payload_hash)
--   embed -> clean.products (rows where embedded_hash <> search_text_hash)
--
-- Safe to run more than once (IF NOT EXISTS everywhere).
-- Requires ParadeDB 0.26+ for the search-index syntax in section 4.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector: embeddings + HNSW
CREATE EXTENSION IF NOT EXISTS pg_search;   -- ParadeDB: BM25 full-text index

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS clean;


-- =============================================================================
-- 1. raw.fetch_runs: one row per fetcher run
-- =============================================================================
-- status:
--   running  : in progress (or crashed without finishing; finished_at stays NULL)
--   complete : reached the empty page with no errors -> catalog is whole,
--              so products missing from this run may be soft-deleted
--   partial  : stopped at max_pages before the end -> never soft-delete from this
--   failed   : FetchError -> never soft-delete from this

CREATE TABLE IF NOT EXISTS raw.fetch_runs (
    run_id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source            text        NOT NULL,                -- e.g. https://kith.com
    started_at        timestamptz NOT NULL DEFAULT now(),
    finished_at       timestamptz,
    status            text        NOT NULL DEFAULT 'running'
                      CHECK (status IN ('running', 'complete', 'partial', 'failed')),
    pages_fetched     integer     NOT NULL DEFAULT 0 CHECK (pages_fetched >= 0),
    products_seen     integer     NOT NULL DEFAULT 0 CHECK (products_seen >= 0),
    products_new      integer     NOT NULL DEFAULT 0 CHECK (products_new >= 0),
    products_changed  integer     NOT NULL DEFAULT 0 CHECK (products_changed >= 0),
    error             text,

    CONSTRAINT fetch_runs_finished_matches_status
        CHECK ((status = 'running') = (finished_at IS NULL))
);

-- "Latest complete run" is what the parser's soft-delete step asks for.
CREATE INDEX IF NOT EXISTS fetch_runs_complete_idx
    ON raw.fetch_runs (source, run_id DESC) WHERE status = 'complete';


-- =============================================================================
-- 2. raw.products: latest raw JSON per product
-- =============================================================================
-- One row per Shopify product. The fetcher upserts on product_id and only
-- rewrites payload / last_changed_* when payload_hash differs.
-- parsed_hash records which payload the parser last processed, so the parser
-- picks up exactly the rows where parsed_hash IS DISTINCT FROM payload_hash.

CREATE TABLE IF NOT EXISTS raw.products (
    product_id          bigint      PRIMARY KEY,           -- Shopify product id
    handle              text        NOT NULL,
    payload             jsonb       NOT NULL,              -- product object from products.json
    payload_hash        text        NOT NULL CHECK (payload_hash ~ '^[0-9a-f]{64}$'),  -- sha256 hex
    shopify_updated_at  timestamptz,

    first_seen_run      bigint      NOT NULL REFERENCES raw.fetch_runs (run_id),
    last_seen_run       bigint      NOT NULL REFERENCES raw.fetch_runs (run_id),
    last_changed_run    bigint      NOT NULL REFERENCES raw.fetch_runs (run_id),
    first_seen_at       timestamptz NOT NULL DEFAULT now(),
    last_seen_at        timestamptz NOT NULL DEFAULT now(),
    last_changed_at     timestamptz NOT NULL DEFAULT now(),

    -- Parser bookkeeping
    parsed_hash         text        CHECK (parsed_hash ~ '^[0-9a-f]{64}$'),
    parsed_at           timestamptz,
    parse_status        text        CHECK (parse_status IN ('ok', 'failed')),

    CONSTRAINT raw_products_parse_fields_together
        CHECK ((parsed_hash IS NULL) = (parse_status IS NULL))
);

-- Parser work queue: only rows whose current payload hasn't been parsed yet.
CREATE INDEX IF NOT EXISTS raw_products_unparsed_idx
    ON raw.products (product_id)
    WHERE parsed_hash IS DISTINCT FROM payload_hash;

-- Soft-delete step: "which products were not seen in run N?"
CREATE INDEX IF NOT EXISTS raw_products_last_seen_run_idx
    ON raw.products (last_seen_run);


-- =============================================================================
-- 3. clean.products / clean.variants / clean.parse_failures
-- =============================================================================
-- clean.products mirrors CleanProduct.product_row() (computed fields included),
-- plus soft-delete state and the embedding.

CREATE TABLE IF NOT EXISTS clean.products (
    product_id            bigint      PRIMARY KEY
                          REFERENCES raw.products (product_id) ON DELETE CASCADE,

    -- Identity and links
    handle                text        NOT NULL UNIQUE,
    url                   text        NOT NULL,
    image_url             text,

    -- Parsed attributes (NULL when the parser couldn't find them)
    title                 text        NOT NULL,
    brand                 text,
    collaborator          text,
    collaborator_note     text,
    model                 text,
    colorway              text,
    style_code            text,
    product_type          text,
    category              text,
    subcategory           text,
    gender                text        CHECK (gender IN ('mens', 'womens', 'unisex', 'kids')),
    materials             text[]      NOT NULL DEFAULT '{}',
    description_bullets   text[]      NOT NULL DEFAULT '{}',
    clean_tags            text[]      NOT NULL DEFAULT '{}',
    purchase_limit        integer     CHECK (purchase_limit >= 1),

    -- Commerce fields, rolled up from variants (filters, not search text)
    price_min             numeric(10, 2) NOT NULL CHECK (price_min >= 0),
    price_max             numeric(10, 2) NOT NULL,
    compare_at_price      numeric(10, 2),
    on_sale               boolean     NOT NULL DEFAULT false,
    in_stock              boolean     NOT NULL DEFAULT false,
    sizes_available       text[]      NOT NULL DEFAULT '{}',

    -- Search document
    search_text           text        NOT NULL,
    search_text_hash      text        NOT NULL CHECK (search_text_hash ~ '^[0-9a-f]{64}$'),

    -- Timestamps
    shopify_created_at    timestamptz,
    shopify_updated_at    timestamptz NOT NULL,
    shopify_published_at  timestamptz,
    scraped_at            timestamptz NOT NULL,
    parsed_at             timestamptz NOT NULL DEFAULT now(),

    -- Soft delete: set false when a *complete* fetch run no longer sees the product
    is_active             boolean     NOT NULL DEFAULT true,
    deactivated_at        timestamptz,

    -- Embedding (filled by the embed step).
    -- 768 dims: request this output size from the Google embedding model
    -- (its 3072 default exceeds pgvector's 2000-dim HNSW limit).
    -- To change it, alter this column and rebuild the HNSW index.
    embedding             vector(768),
    embedding_model       text,
    embedded_hash         text        CHECK (embedded_hash ~ '^[0-9a-f]{64}$'),
    embedded_at           timestamptz,

    CONSTRAINT products_price_range        CHECK (price_max >= price_min),
    CONSTRAINT products_sale_has_original  CHECK (on_sale = (compare_at_price IS NOT NULL)),
    CONSTRAINT products_deactivated_state  CHECK (is_active = (deactivated_at IS NULL)),
    CONSTRAINT products_embedding_together CHECK (
        (embedding IS NULL) = (embedded_hash IS NULL)
        AND (embedding IS NULL) = (embedding_model IS NULL)
    )
);

CREATE TABLE IF NOT EXISTS clean.variants (
    variant_id        bigint      PRIMARY KEY,
    product_id        bigint      NOT NULL
                      REFERENCES clean.products (product_id) ON DELETE CASCADE,
    size              text,
    sku               text,
    price             numeric(10, 2) NOT NULL CHECK (price >= 0),
    compare_at_price  numeric(10, 2),
    on_sale           boolean     NOT NULL DEFAULT false,
    available         boolean     NOT NULL,

    CONSTRAINT variants_sale_has_original CHECK (on_sale = (compare_at_price IS NOT NULL)),
    CONSTRAINT variants_discount_is_real  CHECK (compare_at_price IS NULL OR compare_at_price > price)
);

-- Latest parse failure per product; the parser deletes the row on success.
CREATE TABLE IF NOT EXISTS clean.parse_failures (
    product_id    bigint      PRIMARY KEY
                  REFERENCES raw.products (product_id) ON DELETE CASCADE,
    payload_hash  text        NOT NULL,
    title         text,
    error         text        NOT NULL,
    failed_at     timestamptz NOT NULL DEFAULT now()
);


-- =============================================================================
-- 4. Search indexes
-- =============================================================================

-- 4a. ParadeDB full-text (BM25) index. Only one ParadeDB index is allowed per table.
--   search_text, title : English stemming ("sneaker" matches "sneakers") and
--                        English stopwords ("for", "the" are not indexed, so
--                        "Kith for Wilson" doesn't match every query with "for")
--   brand, model, ...  : default unicode tokenizer (lowercased words)
--   category, gender,
--   clean_tags         : literal (exact values), for filters inside the search
--   numeric / boolean  : included so price/stock filters run inside the index
CREATE INDEX IF NOT EXISTS products_search_idx ON clean.products
USING paradedb (
    product_id,
    (search_text::pdb.simple('stemmer=english', 'stopwords_language=english')),
    (title::pdb.simple('stemmer=english', 'stopwords_language=english')),
    brand,
    collaborator,
    model,
    colorway,
    (category::pdb.literal),
    (subcategory::pdb.literal),
    (gender::pdb.literal),
    (clean_tags::pdb.literal),
    price_min,
    price_max,
    on_sale,
    in_stock,
    is_active
);

-- 4b. Vector (semantic) index. Cosine distance; m / ef_construction are
--     pgvector's defaults, fine for a catalog of this size.
CREATE INDEX IF NOT EXISTS products_embedding_hnsw_idx
    ON clean.products USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 4c. Ordinary B-tree indexes for filters and pipeline queues.
CREATE INDEX IF NOT EXISTS products_active_stock_price_idx
    ON clean.products (is_active, in_stock, price_min);
CREATE INDEX IF NOT EXISTS products_brand_idx
    ON clean.products (lower(brand));
CREATE INDEX IF NOT EXISTS products_category_idx
    ON clean.products (category, subcategory);

-- Embed-step work queue: rows whose search text changed since they were embedded.
CREATE INDEX IF NOT EXISTS products_needs_embedding_idx
    ON clean.products (product_id)
    WHERE embedded_hash IS DISTINCT FROM search_text_hash;

CREATE INDEX IF NOT EXISTS variants_product_idx
    ON clean.variants (product_id);


-- =============================================================================
-- 5. View the search layer queries
-- =============================================================================
CREATE OR REPLACE VIEW clean.searchable_products AS
SELECT *
FROM clean.products
WHERE is_active AND embedding IS NOT NULL;