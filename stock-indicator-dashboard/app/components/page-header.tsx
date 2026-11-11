import type { ReactNode } from "react";

/** Shared page title block: section label, title, one-line description and actions. */
export function PageHeader({ eyebrow, title, description, actions, meta }: {
  eyebrow?: string;
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
  meta?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div className="page-header-main">
        {eyebrow ? <p className="page-eyebrow">{eyebrow}</p> : null}
        <h1>{title}</h1>
        {description ? <p className="page-description">{description}</p> : null}
      </div>
      {actions ? <div className="page-actions">{actions}</div> : null}
      {meta ? <div className="page-meta">{meta}</div> : null}
    </header>
  );
}
