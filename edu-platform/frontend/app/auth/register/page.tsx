"use client";

import { useState } from 'react';

export default function RegisterPage() {
  const [email, setEmail] = useState('');
  const [fullName, setFullName] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('student');
  const [message, setMessage] = useState('');

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, full_name: fullName, password, role }),
      });
      const data = await response.json();
      if (!response.ok) {
        setMessage(data.detail || 'Registration failed');
        return;
      }
      setMessage('Registration successful. Please login.');
    } catch (err) {
      setMessage('Registration error');
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-20 text-slate-100">
      <section className="mx-auto max-w-md rounded-3xl border border-slate-800 bg-slate-900/90 p-10 shadow-2xl shadow-slate-950/40">
        <h1 className="text-3xl font-semibold">Create an account</h1>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div>
            <label className="block text-sm text-slate-300">Full Name</label>
            <input value={fullName} onChange={(e) => setFullName(e.target.value)} className="mt-2 w-full rounded-2xl border border-slate-700 bg-slate-950 px-4 py-3 text-slate-100 outline-none focus:border-cyan-500" />
          </div>
          <div>
            <label className="block text-sm text-slate-300">Email</label>
            <input value={email} onChange={(e) => setEmail(e.target.value)} className="mt-2 w-full rounded-2xl border border-slate-700 bg-slate-950 px-4 py-3 text-slate-100 outline-none focus:border-cyan-500" />
          </div>
          <div>
            <label className="block text-sm text-slate-300">Password</label>
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="mt-2 w-full rounded-2xl border border-slate-700 bg-slate-950 px-4 py-3 text-slate-100 outline-none focus:border-cyan-500" />
          </div>
          <div>
            <label className="block text-sm text-slate-300">Role</label>
            <select value={role} onChange={(e) => setRole(e.target.value)} className="mt-2 w-full rounded-2xl border border-slate-700 bg-slate-950 px-4 py-3 text-slate-100 outline-none focus:border-cyan-500">
              <option value="student">Student</option>
              <option value="teacher">Teacher</option>
              <option value="parent">Parent</option>
            </select>
          </div>
          {message && <p className="text-sm text-slate-300">{message}</p>}
          <button type="submit" className="w-full rounded-2xl bg-cyan-500 px-4 py-3 text-slate-950 transition hover:bg-cyan-400">Register</button>
        </form>
      </section>
    </main>
  );
}
