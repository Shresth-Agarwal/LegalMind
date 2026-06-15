import { useState, useEffect } from 'react'

function SourceList({ sources }) {
  const [open, setOpen] = useState(false)

  function cleanSource(source) {
    return source
      .replace(/_/g, ' ')
      .replace(/\.PDF$|\.pdf$/, '')
      .replace(/ on \d{1,2} \w+ \d{4}$/, '')
      .trim()
  }

  return (
    <div className="border border-gray-800 rounded-xl overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-5 py-4 text-sm text-gray-400 hover:bg-gray-900 transition"
      >
        <span className="font-medium text-gray-300">Indexed Documents</span>
        <span className="text-gray-600">{open ? '▲ hide' : `▼ show ${sources.length} documents`}</span>
      </button>

      {open && (
        <div className="divide-y divide-gray-800">
          {sources.map((src, i) => (
            <div key={i} className="px-5 py-3 text-sm text-gray-400 hover:bg-gray-900 transition">
              <span className="text-indigo-400 font-mono text-xs mr-3">{String(i + 1).padStart(2, '0')}</span>
              {cleanSource(src)}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const [useReranker, setUseReranker] = useState(true)
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [stats, setStats] = useState(null)
  const [searched, setSearched] = useState(false)

  useEffect(() => {
    fetch('/RAG/stats')
      .then(r => r.json())
      .then(setStats)
      .catch(() => {})
  }, [])

  async function handleSearch() {
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    setSearched(true)

    try {
      const res = await fetch('/RAG/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query, top_k: 5, use_reranker: useReranker })
      })
      const data = await res.json()
      setResults(data.results)
    } catch {
      setError('Could not reach the backend. Make sure FastAPI is running on port 8000.')
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter') handleSearch()
  }

  function cleanSource(source) {
    return source
      .replace(/_/g, ' ')
      .replace(/\.PDF$|\.pdf$/, '')
      .replace(/on \d{1,2} \w+ \d{4}$/, '')
      .trim()
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 font-sans">

      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight">⚖️ LegalMind</h1>
          <p className="text-xs text-gray-500 mt-0.5">Hybrid RAG over Indian Legal Documents</p>
        </div>
        {stats && (
          <div className="text-right text-xs text-gray-500">
            <span className="text-gray-400 font-medium">{stats.total_chunks.toLocaleString()}</span> chunks &nbsp;·&nbsp;
            <span className="text-gray-400 font-medium">{stats.unique_sources}</span> documents
          </div>
        )}
      </header>

      {/* Main */}
      <main className="max-w-3xl mx-auto px-6 py-12">

        {/* Search */}
        <div className="mb-8">
          <div className="flex gap-3 mb-4">
            <input
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="e.g. What is the rarest of rare doctrine for death penalty?"
              className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-sm text-gray-100 placeholder-gray-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition"
            />
            <button
              onClick={handleSearch}
              disabled={loading}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium px-6 py-3 rounded-lg transition"
            >
              {loading ? 'Searching…' : 'Search'}
            </button>
          </div>

          {/* Mode toggle */}
          <div className="flex gap-6 text-sm">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="mode"
                checked={!useReranker}
                onChange={() => setUseReranker(false)}
                className="accent-indigo-500"
              />
              <span className="text-gray-400">Hybrid only <span className="text-gray-600">(RRF)</span></span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="mode"
                checked={useReranker}
                onChange={() => setUseReranker(true)}
                className="accent-indigo-500"
              />
              <span className="text-gray-400">Hybrid + Reranker <span className="text-gray-600">(cross-encoder)</span></span>
            </label>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-950 border border-red-800 text-red-300 text-sm rounded-lg px-4 py-3 mb-6">
            {error}
          </div>
        )}

        {/* Results */}
        {results.length > 0 && (
          <div className="space-y-4">
            <p className="text-xs text-gray-500 uppercase tracking-widest mb-4">
              {results.length} results · {useReranker ? 'hybrid + reranker' : 'hybrid (RRF)'}
            </p>
            {results.map(r => (
              <div
                key={r.rank}
                className="bg-gray-900 border border-gray-800 rounded-xl p-5 hover:border-gray-700 transition"
              >
                <div className="flex items-start justify-between gap-4 mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-indigo-400 bg-indigo-950 border border-indigo-900 rounded px-2 py-0.5">
                      #{r.rank}
                    </span>
                    <span className="text-sm font-medium text-gray-200 leading-tight">
                      {cleanSource(r.source)}
                    </span>
                  </div>
                  {r.rerank_score != null && (
                    <span className="text-xs text-gray-500 shrink-0">
                      score: <span className="text-gray-300 font-mono">{r.reranker_score.toFixed(3)}</span>
                    </span>
                  )}
                </div>
                <p className="text-sm text-gray-400 leading-relaxed line-clamp-4">
                  {r.text}
                </p>
              </div>
            ))}
          </div>
        )}

        {/* Empty state */}
        {searched && !loading && results.length === 0 && !error && (
          <p className="text-center text-gray-500 text-sm mt-12">No results found.</p>
        )}

        {/* Initial state */}
        {!searched && (
          <div className="mt-12 space-y-10">
            {/* Sample queries */}
            <div className="text-center text-gray-600 text-sm space-y-2">
              <p>Try queries like:</p>
              <p className="text-gray-500">"What is the rarest of rare doctrine?"</p>
              <p className="text-gray-500">"Section 302 punishment for murder"</p>
              <p className="text-gray-500">"Arrest guidelines under 498A"</p>
            </div>

            {/* Corpus sources */}
            {stats?.sources && (
              <SourceList sources={stats.sources} />
            )}
          </div>
        )}
      </main>
    </div>
  )
}