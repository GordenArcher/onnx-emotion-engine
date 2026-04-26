import axios from "axios";
import { useEffect } from "react";

export function useServerPing(intervalMs = 120_000) {
  useEffect(() => {
    const ping = () => {
      axios.get("/api/v1/health");
    };

    ping();
    const id = setInterval(ping, intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);
}
