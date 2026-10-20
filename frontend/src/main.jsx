import { lazy, StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import './index.css'
import Landing from './pages/Landing'
import Modules from './pages/Modules'
import NotFound from './pages/NotFound'

// The workspace pulls in Recharts + KaTeX; load it only when a module is opened.
// (Layout wraps routes in <Suspense>, so the nav stays visible while it loads.)
const Workspace = lazy(() => import('./pages/Workspace'))
const About = lazy(() => import('./pages/About'))

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Landing />} />
          <Route path="modules" element={<Modules />} />
          <Route path="modules/:id" element={<Workspace />} />
          <Route path="about" element={<About />} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
