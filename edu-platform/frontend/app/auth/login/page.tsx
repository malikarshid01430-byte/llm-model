"use client";

import { useState } from 'react';
import { useDispatch } from 'react-redux';
import { setUser } from '../../store/userSlice';
import { storeToken } from '../../lib/api';

export default function LoginPage() {
  const dispatch = useDispatch();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: email, password }),
      });
      const data = await response.json();
      if (!response.ok) {
        setError(data.detail || 'Login failed');
        return;
      }
      storeToken(data.access_token);
      dispatch(setUser({ id: 'current', email, fullName: '', role: 'student', token: data.access_token }));
      setError('');
    } catch (err) {
      setError('Login error');
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-20 text-slate-100">
      <section className="mx-auto max-w-md rounded-3xl border border-slate-800 bg-slate-900/90 p-10 shadow-2xl shadow-slate-950/40">
        <h1 className="text-3xl font-semibold">Login to EduAI</h1>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div>
            <label className="block text-sm text-slate-300">Email</label>
            <input value={email} onChange={(e) => setEmail(e.target.value)} className="mt-2 w-full rounded-2xl border border-slate-700 bg-slate-950 px-4 py-3 text-slate-100 outline-none focus:border-cyan-500" />
          </div>
          <div>
            <label className="block text-sm text-slate-300">Password</label>
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="mt-2 w-full rounded-2xl border border-slate-700 bg-slate-950 px-4 py-3 text-slate-100 outline-none focus:border-cyan-500" />
          </div>
          {error && <p className="text-sm text-rose-400">{error}</p>}
          <button type="submit" className="w-full rounded-2xl bg-cyan-500 px-4 py-3 text-slate-950 transition hover:bg-cyan-400">Sign In</button>
        </form>
      </section>
    </main>
  );
}
