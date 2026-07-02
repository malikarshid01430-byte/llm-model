"use client";

import Link from 'next/link';

const modules = [
  { title: 'Student Workspace', href: '/dashboard/student', description: 'Courses, assignments, progress and AI tutoring.' },
  { title: 'Teacher Workspace', href: '/dashboard/teacher', description: 'Lesson planning, grading, and analytics.' },
  { title: 'Admin Console', href: '/dashboard/admin', description: 'Institution oversight, users, reports and governance.' },
];

export default function DashboardPage() {
  return (
    <main className="min-h-screen bg-slate-950 px-4 py-10 text-slate-100">
      <section className="mx-auto max-w-6xl rounded-3xl border border-slate-800 bg-slate-900/80 p-8 shadow-2xl shadow-slate-950/40">
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="text-sm uppercase tracking-[0.3em] text-cyan-400">Enterprise Dashboard</p>
            <h1 className="mt-2 text-3xl font-semibold">Choose your role-based workspace</h1>
          </div>
          <Link href="/" className="rounded-full border border-slate-700 px-4 py-2 text-sm text-slate-300 transition hover:border-cyan-500 hover:text-white">
            Back to home
          </Link>
        </div>
        <div className="mt-8 grid gap-6 md:grid-cols-3">
          {modules.map((module) => (
            <Link key={module.title} href={module.href} className="rounded-3xl border border-slate-800 bg-slate-950/80 p-6 transition hover:border-cyan-500">
              <h2 className="text-xl font-semibold">{module.title}</h2>
              <p className="mt-3 text-sm text-slate-400">{module.description}</p>
            </Link>
          ))}
        </div>
      </section>
    </main>
  );
}
