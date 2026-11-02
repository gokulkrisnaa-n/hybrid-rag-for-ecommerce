import { describe, expect, it } from "vitest";

import type { ProductCard } from "./api/types";
import { filterChips, formatPrice, priceView, shortId, sizesText } from "./format";

const card = (over: Partial<ProductCard>): ProductCard => ({
  product_id: 1, title: "T", brand: null, collaborator: null, model: null, colorway: null, category: null,
  subcategory: null, gender: null, price_min: "130.00", price_max: "130.00", compare_at_price: null,
  on_sale: false, in_stock: true, sizes_available: [], url: "https://x", image_url: null, ...over,
});

describe("prices", () => {
  it("formats with cents and thousands separators", () => {
    expect(formatPrice("1495")).toBe("$1,495.00");
    expect(formatPrice(300, { cents: false })).toBe("$300");
    expect(formatPrice(99.5, { cents: false })).toBe("$99.50");
  });

  it("shows the sale price, the old price and the discount", () => {
    expect(priceView(card({ price_min: "97.50", price_max: "97.50", compare_at_price: "130.00", on_sale: true })))
      .toEqual({ current: "$97.50", was: "$130.00", discountPercent: 25 });
  });

  it("shows a range when variants are priced differently", () => {
    expect(priceView(card({ price_min: "90.00", price_max: "110.00" })).current).toBe("$90.00 – $110.00");
  });

  it("ignores a compare-at price that is not a real discount", () => {
    expect(priceView(card({ compare_at_price: "130.00", on_sale: true })).was).toBeNull();
    expect(priceView(card({ compare_at_price: "150.00", on_sale: false })).was).toBeNull();
  });
});

describe("sizesText", () => {
  it("lists short runs and cuts long ones", () => {
    expect(sizesText(["S", "M", "L"])).toBe("S · M · L");
    expect(sizesText(["7", "8", "9", "10", "11", "12", "13", "14"], 6)).toBe("7 · 8 · 9 · 10 · 11 · 12 +2");
  });
});

describe("filterChips", () => {
  it("turns applied filters into readable chips in a stable order", () => {
    const chips = filterChips({
      exclude_brands: ["The North Face"], max_price: "300", categories: ["outerwear"], genders: ["mens"],
      on_sale_only: true,
    });
    expect(chips).toEqual([
      { text: "Outerwear" }, { text: "Men" }, { text: "Under $300" }, { text: "On sale" },
      { text: "Not The North Face", negative: true },
    ]);
  });

  it("shows a price range and a minimum", () => {
    expect(filterChips({ min_price: 50, max_price: 120 })).toEqual([{ text: "$50 – $120" }]);
    expect(filterChips({ min_price: 50 })).toEqual([{ text: "From $50" }]);
  });

  it("ignores unknown and empty values", () => {
    expect(filterChips({ in_stock_only: true, something_new: "x", brands: [] })).toEqual([]);
    expect(filterChips(undefined)).toEqual([]);
  });
});

it("shortens request ids", () => {
  expect(shortId("7f3a9c1e0b2d4e6f8a")).toBe("7f3a9c1e…");
  expect(shortId("abc")).toBe("abc");
});