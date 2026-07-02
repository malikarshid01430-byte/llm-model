"use client";

import { useEffect, useState } from 'react';

import { apiFetch } from '../../lib/api';

export default function AdminDashboardPage() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiFetch('/api/admin/dashboard')
      .then(setData)
      .catch(() => {
        setError('Unable to load admin dashboard');
        setData(null);
      });
  }, []);

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-10 text-slate-100">
      <section className="mx-auto max-w-5xl rounded-3xl border border-slate-800 bg-slate-900/80 p-8">
        <h1 className="text-3xl font-semibold">Administration Console</h1>
        <p className="mt-3 text-slate-400">Governance, analytics, reports, and institutional controls.</p>
        <div className="mt-8 rounded-3xl border border-slate-800 bg-slate-950/70 p-6">
          {error ? (
            <p className="text-rose-400">{error}</p>
          ) : data ? (
            <pre className="overflow-x-auto text-sm text-slate-300">{JSON.stringify(data, null, 2)}</pre>
          ) : (
            <p className="text-slate-400">Loading administration metrics...</p>
          )}
        </div>
      </section>
    </main>
  );
}
