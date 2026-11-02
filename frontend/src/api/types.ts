// Shapes of the API's JSON, mirroring app/api/schemas.py and app/generation/schemas.py.
// Prices are Decimals on the server and arrive as strings ("130.00").

export type AnswerKind = "recommend" | "compare" | "no_match" | "clarify" | "out_of_scope" | "error";

export interface ProductCard {
  product_id: number;
  title: string;
  brand: string | null;
  collaborator: string | null;
  model: string | null;
  colorway: string | null;
  category: string | null;
  subcategory: string | null;
  gender: string | null;
  price_min: string;
  price_max: string;
  compare_at_price: string | null;
  on_sale: boolean;
  in_stock: boolean;
  sizes_available: string[];
  url: string;
  image_url: string | null;
}

export interface AnswerProduct {
  product: ProductCard;
  reason: string;
}

export interface ComparisonTable {
  columns: string[];
  rows: { product_id: number; title: string; values: string[] }[];
}

export interface ChatAnswer {
  request_id: string;
  kind: AnswerKind;
  message: string;
  products: AnswerProduct[];
  comparison: ComparisonTable | null;
  follow_up: string | null;
  source: "llm" | "template";
  /** Filters the search used: the shopper's plus those inferred from the message. */
  applied_filters?: Record<string, unknown>;
  warnings: string[];
  timings_ms: Record<string, number>;
  debug: Record<string, unknown> | null;
}

export interface HistoryTurn {
  role: "user" | "assistant";
  content: string;
}

export interface SearchFilters {
  min_price?: number;
  max_price?: number;
  genders?: string[];
  categories?: string[];
  brands?: string[];
  exclude_brands?: string[];
  sizes?: string[];
  on_sale_only?: boolean;
}

export interface ChatRequest {
  message: string;
  history: HistoryTurn[];
  filters?: SearchFilters;
  session_id?: string;
  debug?: boolean;
}

/** The API's single error format: {"error": {code, message, request_id, details}}. */
export interface ErrorBody {
  code: string;
  message: string;
  request_id: string;
  details?: { field: string; message: string }[] | null;
}

/** Stream stages, in the order the server sends them. */
export type Stage = "searching" | "writing";

// Limits enforced by the API (app/api/schemas.py); the client checks them first.
export const MAX_MESSAGE_CHARS = 500;
export const MAX_HISTORY_TURNS = 10;