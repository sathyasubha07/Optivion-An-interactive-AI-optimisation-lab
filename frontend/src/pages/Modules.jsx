import ModuleGrid from '../components/ModuleGrid'
import { FadeIn, SectionTitle } from '../components/ui'

export default function Modules() {
  return (
    <section className="mx-auto max-w-6xl px-4 py-16 sm:px-6">
      <FadeIn>
        <SectionTitle eyebrow="Modules" title="Optimisation labs">
          Open a ready module to walk through the problem, the maths, a live run and the results.
        </SectionTitle>
      </FadeIn>
      <div className="mt-10">
        <ModuleGrid />
      </div>
    </section>
  )
}
