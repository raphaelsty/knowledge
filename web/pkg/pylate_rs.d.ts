/* tslint:disable */
/* eslint-disable */
/**
 * WASM-compatible version of the `hierarchical_pooling` function.
 */
export function hierarchical_pooling(input: any): any;
/**
 * The main ColBERT model structure.
 *
 * This struct encapsulates the language model, a linear projection layer,
 * the tokenizer, and all necessary configuration for performing encoding
 * and similarity calculations based on the ColBERT architecture.
 */
export class ColBERT {
  free(): void;
  /**
   * WASM-compatible constructor.
   */
  constructor(weights: Uint8Array, dense_weights: Uint8Array, tokenizer: Uint8Array, config: Uint8Array, sentence_transformers_config: Uint8Array, dense_config: Uint8Array, special_tokens_map: Uint8Array, batch_size?: number | null);
  /**
   * WASM-compatible version of the `encode` method.
   */
  encode(input: any, is_query: boolean): any;
  /**
   * WASM-compatible version of the `similarity` method.
   */
  similarity(input: any): any;
  /**
   * WASM-compatible method to get the raw similarity matrix and tokens.
   */
  raw_similarity_matrix(input: any): any;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly colbert_from_bytes: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number) => [number, number, number];
  readonly colbert_encode: (a: number, b: any, c: number) => [number, number, number];
  readonly colbert_similarity: (a: number, b: any) => [number, number, number];
  readonly colbert_raw_similarity_matrix: (a: number, b: any) => [number, number, number];
  readonly hierarchical_pooling: (a: any) => [number, number, number];
  readonly __wbg_colbert_free: (a: number, b: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_4: WebAssembly.Table;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
