-- Rebuild the BM25 index with English stopwords on search_text and title.
-- db/schema.sql already has the new definition (for fresh databases); this
-- file updates an existing one. Takes a few seconds for 25k products.
--
--   docker exec -i kith-db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < db/migrations/001_bm25_stopwords.sql

BEGIN;

DROP INDEX IF EXISTS clean.products_search_idx;

CREATE INDEX products_search_idx ON clean.products
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

COMMIT;

-- Check: "for" should now match nothing, "wilson" still should.
SELECT
    (SELECT count(*) FROM clean.products WHERE search_text ||| 'for')    AS for_matches,
    (SELECT count(*) FROM clean.products WHERE search_text ||| 'wilson') AS wilson_matches;
