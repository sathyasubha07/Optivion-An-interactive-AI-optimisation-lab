import { Button } from '../components/ui'

export default function NotFound() {
  return (
    <section className="mx-auto max-w-xl px-4 py-24 text-center">
      <h1 className="font-display text-4xl font-semibold">Page not found</h1>
      <Button to="/" className="mt-8">
        Back home
      </Button>
    </section>
  )
}
