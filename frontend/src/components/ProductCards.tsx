import { useState } from "react";

import type { AnswerProduct } from "../api/types";
import { priceView, sizesText } from "../format";
import "./ProductCards.css";

function Card({ product: p, reason }: AnswerProduct) {
  const [imageFailed, setImageFailed] = useState(false);
  const price = priceView(p);
  const brand = p.collaborator && p.brand ? `${p.brand} × ${p.collaborator}` : (p.brand ?? "");

  return (
    <a className="card" href={p.url} target="_blank" rel="noreferrer">
      <div className="card__image">
        {p.image_url && !imageFailed ? (
          <img src={p.image_url} alt={p.title} loading="lazy" onError={() => setImageFailed(true)} />
        ) : (
          <span className="card__noimage" aria-hidden="true">No image</span>
        )}
        {price.discountPercent !== null && <span className="card__badge card__badge--sale">−{price.discountPercent}%</span>}
        {!p.in_stock && <span className="card__badge">Sold out</span>}
      </div>
      <div className="card__body">
        {brand && <p className="card__brand">{brand}</p>}
        <h3 className="card__title">{p.title}</h3>
        <p className="card__price">
          <span className={price.was ? "card__now card__now--sale" : "card__now"}>{price.current}</span>
          {price.was && <s className="card__was">{price.was}</s>}
        </p>
        {p.sizes_available.length > 0 && <p className="card__sizes">{sizesText(p.sizes_available)}</p>}
        {reason && <p className="card__reason">{reason}</p>}
      </div>
    </a>
  );
}

export function ProductCards({ products }: { products: AnswerProduct[] }) {
  return (
    <div className="cards">
      {products.map((ap) => (
        <Card key={ap.product.product_id} {...ap} />
      ))}
    </div>
  );
}