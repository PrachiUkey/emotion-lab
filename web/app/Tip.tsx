'use client';
import { cloneElement, isValidElement, useId, useState } from 'react';

// Hover (or keyboard focus, or tap on touch screens) to show a plain-language explanation.
// side: which edge the bubble lines up with, so it stays on screen near the page edges.
// interactive: the child is itself a button — show on hover/focus only, leave clicks to the button.
// Screen readers read the term, then the explanation (aria-describedby).
export default function Tip({ text, children, side = 'left', interactive = false }: {
  text: string; children: React.ReactNode; side?: 'left' | 'right'; interactive?: boolean;
}) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const body = <span id={id} className={`tip-body ${side}`} role="tooltip">{text}</span>;
  if (interactive) {
    // describe the focusable child (e.g. the button) itself, not the wrapper
    const child = isValidElement(children)
      ? cloneElement(children as React.ReactElement<{ 'aria-describedby'?: string }>, { 'aria-describedby': id })
      : children;
    return <span className="tip interactive">{child}{body}</span>;
  }
  return (
    <span className={`tip ${open ? 'open' : ''}`} tabIndex={0} aria-describedby={id}
          onClick={() => setOpen((o) => !o)} onBlur={() => setOpen(false)}>
      {children}
      {body}
    </span>
  );
}
