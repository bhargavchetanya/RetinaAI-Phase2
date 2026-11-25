"use client";

import { useEffect, useState } from "react";
import { api, type Manifest } from "@/lib/api";

/** Loads web_models/manifest.json from the server once per page. */
export function useManifest() {
  const [data, setData] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.models().then(setData).catch((e) => setError((e as Error).message));
  }, []);
  return { data, error };
}
