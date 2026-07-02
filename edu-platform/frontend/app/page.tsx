import Link from 'next/link';
import DashboardCard from './components/DashboardCard';

const features = [
  { title: 'AI Tutor', detail: 'Conversational learning and doubt solving.' },
  { title: 'Personalized Paths', detail: 'Adaptive plans for every learner.' },
  { title: 'Analytics', detail: 'Student progress and performance insights.' },
  { title: 'Teacher Tools', detail: 'Create courses, quizzes and automatic grading.' },
];

export default function Home() {
  return (
    <main className="min-h-screen bg-slate-950 px-4 py-10 text-slate-50">
      <section className="mx-auto max-w-6xl">
        <div className="flex flex-col gap-8 rounded-3xl border border-slate-700 bg-slate-900/80 p-10 shadow-2xl shadow-slate-950/50">
          <div>
            <h1 className="text-4xl font-semibold">EduAI Platform</h1>
            <p className="mt-4 max-w-2xl text-slate-300">AI-powered education platform for students, teachers, parents, administrators and institutions.</p>
          </div>
          <div className="grid gap-6 md:grid-cols-2">
            {features.map((feature) => (
              <DashboardCard key={feature.title} title={feature.title} description={feature.detail} />
            ))}
          </div>
          <div className="flex flex-wrap gap-4">
            <Link href="/auth/login" className="rounded-full bg-cyan-500 px-6 py-3 text-slate-950 transition hover:bg-cyan-400">Login</Link>
            <Link href="/auth/register" className="rounded-full border border-slate-600 px-6 py-3 text-slate-50 transition hover:border-slate-400">Register</Link>
            <Link href="/dashboard" className="rounded-full border border-cyan-500/50 px-6 py-3 text-cyan-300 transition hover:border-cyan-400 hover:text-cyan-200">Open Dashboard</Link>
          </div>
        </div>
      </section>
    </main>
  );
}
