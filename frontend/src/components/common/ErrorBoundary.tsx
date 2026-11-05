"use client";

import React, { Component, type ErrorInfo, type ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  children: ReactNode;
  name?: string;
  fallback?: ReactNode;
  onReset?: () => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error(`ErrorBoundary caught an error in ${this.props.name ?? "component"}:`, error, errorInfo);
  }

  public handleReset = () => {
    this.setState({ hasError: false, error: null });
    this.props.onReset?.();
  };

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="hud-glass m-4 flex flex-col items-center justify-center space-y-3 rounded-[20px] p-6 text-center">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-red-500/10 text-red-600 ring-1 ring-inset ring-red-500/30 dark:text-red-400">
            <AlertTriangle aria-hidden="true" className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-neutral-900 dark:text-neutral-100">
              {this.props.name ? `${this.props.name} failed to load` : "Something went wrong"}
            </h3>
            <p className="mt-1 max-w-sm text-xs text-neutral-500 dark:text-neutral-400">
              {this.state.error?.message ?? "An unexpected rendering error occurred."}
            </p>
          </div>
          <button
            type="button"
            onClick={this.handleReset}
            className="hud-btn hud-btn-active"
          >
            <RefreshCw aria-hidden="true" className="h-3.5 w-3.5" />
            <span>Retry</span>
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
