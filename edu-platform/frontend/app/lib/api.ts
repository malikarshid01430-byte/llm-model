const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ||
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  'http://localhost:8000';

const TOKEN_KEY = 'eduai_access_token';

export function getStoredToken(): string | null {
  if (typeof window === 'undefined') {
    return null;
  }
  return window.localStorage.getItem(TOKEN_KEY);
}

export function storeToken(token: string): void {
  if (typeof window === 'undefined') {
    return;
  }
  window.localStorage.setItem(TOKEN_KEY, token);
}

export function clearStoredToken(): void {
  if (typeof window === 'undefined') {
    return;
  }
  window.localStorage.removeItem(TOKEN_KEY);
}

export async function apiFetch(path: string, init: RequestInit = {}) {
  const token = getStoredToken();
  const headers = new Headers(init.headers ?? {});
  headers.set('Accept', 'application/json');
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers,
  });

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  return response.json();
}

export { API_BASE };
