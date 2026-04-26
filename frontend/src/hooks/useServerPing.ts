import { useEffect } from "react";

const API_BASE = import.meta.env.VITE_API_URL;

export function useServerPing(intervalMs = 120_000) {
  useEffect(() => {
    const ping = () => {
      fetch(`${API_BASE}/api/v1/health`).catch(() => {});
    };

    ping();
    const id = setInterval(ping, intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
}
