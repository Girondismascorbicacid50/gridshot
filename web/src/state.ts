import { create } from "zustand";
import type { BatchResult, GenerateParams, Session, TraceResult } from "./api";

const ACTIVE_SESSION_KEY = "gridshot.active-single-session.v1";

export interface ActiveSessionRef {
  session: string;
  params: GenerateParams;
}

function rememberActiveSession(session: Session, params: GenerateParams) {
  try {
    localStorage.setItem(
      ACTIVE_SESSION_KEY,
      JSON.stringify({ session: session.session, params }),
    );
  } catch {
    // Private browsing or storage policy must not block the editor itself.
  }
}

export function loadActiveSession(): ActiveSessionRef | null {
  try {
    const value = JSON.parse(localStorage.getItem(ACTIVE_SESSION_KEY) ?? "null");
    return value?.session && value?.params ? value : null;
  } catch {
    return null;
  }
}

export function forgetActiveSession() {
  try {
    localStorage.removeItem(ACTIVE_SESSION_KEY);
  } catch {
    // Storage is an optional convenience only.
  }
}

export type View =
  | "upload"
  | "editor"
  | "tracing"
  | "result"
  | "library"
  | "batch"
  | "calibration";

interface AppState {
  view: View;
  currentToolView: "editor" | "result" | null;
  session: Session | null;
  params: GenerateParams | null;
  result: TraceResult | null;
  batch: BatchResult | null;
  error: string | null;
  setTracing: () => void;
  setEditor: (s: Session, p: GenerateParams) => void;
  setResult: (r: TraceResult, p?: GenerateParams) => void;
  setLibrary: () => void;
  setBatch: (b: BatchResult | null) => void;
  setCalibration: () => void;
  navigate: (view: Exclude<View, "tracing">) => void;
  navigateCurrentTool: () => void;
  setError: (e: string) => void;
  reset: () => void;
}

export const useApp = create<AppState>((set) => ({
  view: "upload",
  currentToolView: null,
  session: null,
  params: null,
  result: null,
  batch: null,
  error: null,
  setTracing: () => set({ view: "tracing", error: null }),
  setEditor: (session, params) => {
    rememberActiveSession(session, params);
    set({
      view: "editor",
      currentToolView: "editor",
      session,
      params,
      error: null,
    });
  },
  setResult: (result, params) =>
    set((state) => ({
      view: "result",
      currentToolView: "result",
      result,
      params: params ?? state.params,
    })),
  setLibrary: () => set({ view: "library", error: null }),
  setBatch: (b) => set({ view: "batch", batch: b, error: null }),
  setCalibration: () => set({ view: "calibration", error: null }),
  navigate: (view) => set({ view, error: null }),
  navigateCurrentTool: () =>
    set((state) => ({
      view: state.currentToolView
        ?? (state.result ? "result" : state.session ? "editor" : "upload"),
      error: null,
    })),
  setError: (e) => set({ view: "upload", error: e }),
  reset: () => {
    forgetActiveSession();
    set({
      view: "upload",
      currentToolView: null,
      session: null,
      params: null,
      result: null,
      batch: null,
      error: null,
    });
  },
}));
