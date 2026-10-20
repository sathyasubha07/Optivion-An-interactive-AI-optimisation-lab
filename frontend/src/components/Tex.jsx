import katex from 'katex'
import 'katex/dist/katex.min.css'
import { useMemo } from 'react'

export default function Tex({ children, display = false, className = '' }) {
  const html = useMemo(
    () =>
      katex.renderToString(String(children ?? ''), {
        displayMode: display,
        throwOnError: false,
        strict: 'ignore',
      }),
    [children, display],
  )
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />
}
