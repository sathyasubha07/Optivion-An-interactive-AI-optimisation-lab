import { AnimatePresence, motion } from 'framer-motion'
import { Suspense, useEffect, useState } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'

function Logo() {
  return (
    <Link to="/" className="flex items-center gap-2.5" aria-label="OptiVision home">
      <img src="/favicon.svg" alt="" className="h-8 w-8" />
      <span className="font-display text-lg font-semibold tracking-tight">
        Opti<span className="text-gradient">Vision</span>
      </span>
    </Link>
  )
}

const links = [
  { to: '/modules', label: 'Modules' },
  { to: '/about', label: 'About' },
]

function Nav() {
  const [open, setOpen] = useState(false)

  return (
    <header className="sticky top-0 z-40 border-b border-white/5 bg-ink-950/70 backdrop-blur-xl">
      <nav className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
        <Logo />
        <div className="hidden items-center gap-1 sm:flex">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              className={({ isActive }) =>
                `rounded-lg px-3 py-2 text-sm transition ${isActive ? 'bg-white/10 text-mist-100' : 'text-mist-300 hover:text-mist-100'}`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </div>
        <button
          type="button"
          className="rounded-lg p-2 text-mist-300 sm:hidden"
          aria-label="Toggle menu"
          aria-expanded={open}
          onClick={() => setOpen((o) => !o)}
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            {open ? <path d="M6 6l12 12M18 6L6 18" /> : <path d="M4 7h16M4 12h16M4 17h16" />}
          </svg>
        </button>
      </nav>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden border-t border-white/5 sm:hidden"
          >
            <div className="flex flex-col px-4 py-2">
              {links.map((l) => (
                <NavLink key={l.to} to={l.to} onClick={() => setOpen(false)} className="rounded-lg px-3 py-3 text-mist-300">
                  {l.label}
                </NavLink>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  )
}

function Background() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden" aria-hidden="true">
      <div className="grid-bg absolute inset-0" />
      <motion.div
        className="absolute -top-40 left-1/2 h-[520px] w-[820px] -translate-x-1/2 rounded-full bg-teal/10 blur-[120px]"
        animate={{ opacity: [0.5, 0.8, 0.5] }}
        transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="absolute right-[-200px] top-[30%] h-[420px] w-[420px] rounded-full bg-violet/10 blur-[110px]"
        animate={{ y: [0, 40, 0] }}
        transition={{ duration: 14, repeat: Infinity, ease: 'easeInOut' }}
      />
    </div>
  )
}

function Footer() {
  return (
    <footer className="mt-24 border-t border-white/5">
      <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-8 text-sm text-mist-500 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <p>OptiVision · MFC course project · Team 08</p>
        <p>Every number on screen comes from a real solver run.</p>
      </div>
    </footer>
  )
}

export default function Layout() {
  const { pathname } = useLocation()
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [pathname])
  return (
    <div className="relative flex min-h-screen flex-col">
      <Background />
      <Nav />
      <main className="flex-1">
        <Suspense
          fallback={
            <section className="mx-auto max-w-6xl px-4 py-16 sm:px-6">
              <div className="glass h-40 animate-pulse rounded-2xl" />
            </section>
          }
        >
          <Outlet />
        </Suspense>
      </main>
      <Footer />
    </div>
  )
}
