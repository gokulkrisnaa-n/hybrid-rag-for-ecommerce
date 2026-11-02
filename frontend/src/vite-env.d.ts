/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Optional, at build time: a Langfuse project URL, e.g. https://cloud.langfuse.com/project/<id>.
   *  The debug panel then links each answer to its trace. */
  readonly VITE_LANGFUSE_PROJECT_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}