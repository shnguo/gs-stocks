import { getChatGPTUser } from "../chatgpt-auth";

export async function marketDataSubject(): Promise<string> {
  const user = await getChatGPTUser();
  const source = user?.userId ?? "local-preview-user";
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(source));
  const value = Array.from(
    new Uint8Array(digest),
    (byte) => byte.toString(16).padStart(2, "0"),
  ).join("").slice(0, 32);
  return `dashboard:${value}`;
}
