import type { AnswerProduct, ComparisonTable as Table } from "../api/types";
import "./ComparisonTable.css";

interface Props {
  table: Table;
  products: AnswerProduct[];
}

/**
 * Products side by side: one column per product, one row per attribute. The last
 * rows (price, sale, sizes) are filled from the catalog by the server, not the model.
 */
export function ComparisonTable({ table, products }: Props) {
  const urls = new Map(products.map((p) => [p.product.product_id, p.product.url]));
  return (
    <div className="compare" role="region" aria-label="Comparison" tabIndex={0}>
      <table>
        <thead>
          <tr>
            <th scope="col" className="compare__corner" />
            {table.rows.map((r) => (
              <th scope="col" key={r.product_id}>
                {urls.has(r.product_id) ? (
                  <a href={urls.get(r.product_id)} target="_blank" rel="noreferrer">{r.title}</a>
                ) : (
                  r.title
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.columns.map((attribute, i) => (
            <tr key={attribute}>
              <th scope="row">{attribute}</th>
              {table.rows.map((r) => (
                <td key={r.product_id} className={r.values[i] === "Not stated" ? "compare__na" : undefined}>
                  {r.values[i] ?? "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}