/**
 * A small Server-Sent Events parser for streams read with fetch().
 *
 * EventSource only supports GET, and the chat stream is a POST, so the response
 * body is read as text chunks and fed in here. Chunks arrive at arbitrary byte
 * boundaries: an event, a line, even a "\r\n" pair can be split across two
 * chunks, so everything incomplete is kept until the next feed().
 *
 * Follows the WHATWG event-stream rules that matter here: lines end in \n, \r\n
 * or \r; a blank line dispatches the event; several data: lines are joined with
 * \n; lines starting with ":" are comments; one leading space after the colon is
 * dropped; an event with no data is not dispatched.
 */

export interface SSEMessage {
  event: string;
  data: string;
}

export class SSEParser {
  private buffer = "";
  private event = "";
  private data: string[] = [];

  /** Add a chunk of text; returns the events it completed (possibly none). */
  feed(chunk: string): SSEMessage[] {
    this.buffer += chunk;
    const out: SSEMessage[] = [];
    let start = 0;
    for (let i = 0; i < this.buffer.length; i++) {
      const ch = this.buffer[i];
      if (ch !== "\n" && ch !== "\r") continue;
      // A "\r" at the very end may be the first half of "\r\n": wait for more.
      if (ch === "\r" && i === this.buffer.length - 1) break;
      this.line(this.buffer.slice(start, i), out);
      if (ch === "\r" && this.buffer[i + 1] === "\n") i++;
      start = i + 1;
    }
    this.buffer = this.buffer.slice(start);
    return out;
  }

  /** End of stream: a final event without its blank line is dropped, as the spec says. */
  reset(): void {
    this.buffer = "";
    this.event = "";
    this.data = [];
  }

  private line(line: string, out: SSEMessage[]): void {
    if (line === "") {
      if (this.data.length) out.push({ event: this.event || "message", data: this.data.join("\n") });
      this.event = "";
      this.data = [];
      return;
    }
    if (line.startsWith(":")) return;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") this.event = value;
    else if (field === "data") this.data.push(value);
    // "id" and "retry" are not used by this API.
  }
}