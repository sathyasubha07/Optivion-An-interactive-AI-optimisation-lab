import { useCallback, useEffect, useState } from 'react'

const cache = new Map()

// Small fetch hook with a per-key cache, so navigating between pages does not refetch.
export default function useAsync(key, fn) {
  const [state, setState] = useState(() =>
    cache.has(key) ? { data: cache.get(key), error: null, loading: false } : { data: null, error: null, loading: true },
  )

  const load = useCallback(() => {
    setState((s) => ({ ...s, loading: true, error: null }))
    fn()
      .then((data) => {
        cache.set(key, data)
        setState({ data, error: null, loading: false })
      })
      .catch((error) => setState({ data: null, error, loading: false }))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  useEffect(() => {
    if (!cache.has(key)) load()
    else setState({ data: cache.get(key), error: null, loading: false })
  }, [key, load])

  return { ...state, reload: load }
}
