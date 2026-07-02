"use client";

import { useEffect, useState } from 'react';

import { apiFetch } from '../../lib/api';

export default function StudentDashboardPage() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiFetch('/api/student/dashboard')
      .then(setData)
      .catch(() => {
        setError('Unable to load student dashboard');
        setData(null);
      });
  }, []);

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-10 text-slate-100">
      <section className="mx-auto max-w-5xl rounded-3xl border border-slate-800 bg-slate-900/80 p-8">
        <h1 className="text-3xl font-semibold">Student Workspace</h1>
        <p className="mt-3 text-slate-400">Courses, homework, attendance, and AI tutoring tools.</p>
        <div className="mt-8 grid gap-6 md:grid-cols-2">
          <div className="rounded-3xl border border-slate-800 bg-slate-950/70 p-6">
            <h2 className="text-xl font-semibold">Learning Snapshot</h2>
            {error ? (
              <p className="mt-4 text-rose-400">{error}</p>
            ) : data ? (
              <pre className="mt-4 overflow-x-auto text-sm text-slate-300">{JSON.stringify(data, null, 2)}</pre>
            ) : (
              <p className="mt-4 text-slate-400">Loading student dashboard...</p>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}
