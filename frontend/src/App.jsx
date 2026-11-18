
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  ArrowUpRight, BookOpen, Check, ChevronDown, ChevronUp,
  CircleAlert, Copy, Database, Download, FileJson, FileText,
  Filter, LoaderCircle, Menu, MessageSquareText, Plus, Search,
  Send, Settings2, Trash2, X,
} from "lucide-react";

const API_ROUTE = "/graphrag/ask";
const DB_NAME = "parliamentai-conversations";
const DB_STORE = "app";
const DB_KEY = "workspace-v1";

const QUESTIONS = [
  "What were the main arguments for leaving the European Union?",
  "How did Labour and Conservative positions on Brexit differ?",
  "What was the Northern Ireland backstop and why was it controversial?",
  "How did the customs union debate evolve from 2016 to 2019?",
];

// ============================================================
// PERSISTENT STORAGE — SAME DATABASE AS V3
// ============================================================

let databasePromise;
let writeQueue = Promise.resolve();

function openDatabase() {
  if (!databasePromise) {
    databasePromise = new Promise((resolve, reject) => {
      if (!window.indexedDB) {
        reject(new Error("IndexedDB is unavailable."));
        return;
      }

      const request = window.indexedDB.open(DB_NAME, 1);

      request.onupgradeneeded = () => {
        if (!request.result.objectStoreNames.contains(DB_STORE)) {
          request.result.createObjectStore(DB_STORE);
        }
      };

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      request.onblocked = () =>
        reject(new Error("History database is blocked."));
    }).catch((error) => {
      databasePromise = null;
      throw error;
    });
  }

  return databasePromise;
}

async function loadHistory() {
  const db = await openDatabase();

  return new Promise((resolve, reject) => {
    const request = db.transaction(DB_STORE, "readonly")
      .objectStore(DB_STORE).get(DB_KEY);

    request.onsuccess = () => resolve(request.result || null);
    request.onerror = () => reject(request.error);
  });
}

function saveHistory(value) {
  // Serialize writes so older snapshots cannot finish
  // after newer ones.

  const write = async () => {
    const db = await openDatabase();

    return new Promise((resolve, reject) => {
      const transaction = db.transaction(DB_STORE, "readwrite");

      transaction.objectStore(DB_STORE).put(value, DB_KEY);

      transaction.oncomplete = resolve;
      transaction.onerror = () => reject(transaction.error);
      transaction.onabort = () => reject(transaction.error);
    });
  };

  writeQueue = writeQueue.catch(() => {}).then(write);

  return writeQueue;
}

function makeId() {
  return window.crypto?.randomUUID?.() ||
    `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function newConversation() {
  const now = new Date().toISOString();

  return {
    id: makeId(),
    title: "New conversation",
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

function makeTitle(question) {
  const short = question.replace(/\s+/g, " ").trim();

  return short.length > 53
    ? `${short.slice(0, 53)}…`
    : short;
}

// ============================================================
// EXPORT — FULL ANSWERS AND ORIGINAL EVIDENCE
// ============================================================

function safeFilename(name) {
  const slug = String(name || "research")
    .normalize("NFKD")
    .replace(/[^\x20-\x7E]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 55);

  return slug || "research";
}

function downloadText(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;
  link.download = filename;

  document.body.appendChild(link);
  link.click();
  link.remove();

  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function citedIndices(answer) {
  return new Set(
    [...String(answer || "").matchAll(/\[E(\d+)\]/g)]
      .map((match) => Number(match[1]) - 1)
  );
}

function asBlockquote(value) {
  return String(value || "")
    .replace(/\r\n?/g, "\n")
    .split("\n")
    .map((line) => `> ${line}`)
    .join("\n");
}

function exportConversationMarkdown(chat) {
  const lines = [
    `# ParliamentAI research: ${chat.title}`,
    "",
    `Created: ${chat.createdAt || "Unknown"}`,
    `Updated: ${chat.updatedAt || "Unknown"}`,
    `Exported: ${new Date().toISOString()}`,
    "",
    "_Parliamentary speech research prototype (2015–2019). " +
      "Citations refer to evidence attached to each answer; " +
      "automated answers require source verification._",
    "",
    "---",
    "",
  ];

  let questionNumber = 0;

  for (const message of chat.messages || []) {
    if (message.role === "user") {
      questionNumber += 1;

      lines.push(
        `## Question ${questionNumber}`,
        "",
        message.content || "",
        ""
      );

      continue;
    }

    if (message.role !== "assistant") {
      continue;
    }

    lines.push(`## Answer ${questionNumber || 1}`, "");

    if (message.error) {
      lines.push(`_Error: ${message.error}_`, "");
    } else if (message.pending) {
      lines.push("_Answer pending at export._", "");
    } else {
      lines.push(
        message.content || "(No answer text)",
        ""
      );
    }

    const evidence = Array.isArray(message.evidence)
      ? message.evidence
      : [];

    if (evidence.length) {
      const cited = citedIndices(message.content);

      lines.push(
        "### Evidence attached to this answer",
        ""
      );

      for (const [index, item] of evidence.entries()) {
        const id = `E${index + 1}`;

        const speaker = String(
          item.speaker || "Unknown speaker"
        ).replace(/\s+/g, " ");

        const party = String(
          item.party || "Unknown party"
        ).replace(/\s+/g, " ");

        const date = String(
          item.date || "Undated"
        ).replace(/\s+/g, " ");

        lines.push(
          `#### [${id}] ${speaker} (${party}, ${date})`,
          "",
          `- Cited in answer: ${
            cited.has(index) ? "Yes" : "No"
          }`,
          `- Agenda: ${
            String(item.agenda || "Unrecorded")
              .replace(/\s+/g, " ")
          }`,
          `- Retrieval source: ${
            String(item.source || "Unrecorded")
          }`,
          "",
          asBlockquote(item.text || ""),
          ""
        );
      }
    }

    lines.push("---", "");
  }

  return lines.join("\n");
}

function exportConversationJSON(chat) {
  return JSON.stringify({
    format: "ParliamentAI conversation export",
    version: 1,
    exportedAt: new Date().toISOString(),
    dataset: "UK parliamentary Brexit speeches, 2015–2019",
    conversation: chat,
  }, null, 2);
}

function exportAllJSON(conversations) {
  return JSON.stringify({
    format: "ParliamentAI conversation backup",
    version: 1,
    exportedAt: new Date().toISOString(),
    dataset: "UK parliamentary Brexit speeches, 2015–2019",
    conversations: conversations.filter(
      (chat) => chat.messages?.length
    ),
  }, null, 2);
}

// ============================================================
// RENDER ANSWERS AND CLICKABLE SOURCE MARKERS
// ============================================================

function readableAnswer(answer) {
  return String(answer || "")
    .replace(
      /\[E(\d+)\](?!\()/g,
      (_, n) => `[E${n}](#source-${n})`
    )
    .replace(
      /(\[E\d+\]\(#source-\d+\)\.)\s+(?=[A-Z][^\n]{0,65}\(20\d{2}-\d{2}-\d{2}\)\s+said:)/g,
      "$1\n\n"
    );
}

function matchingQuote(answer, evidenceNumber, text) {
  let longest = "";

  const pattern =
    /[“"]([^”"]{25,900})[”"]\s*\[E(\d+)\]/g;

  for (const match of String(answer || "").matchAll(pattern)) {
    if (Number(match[2]) !== evidenceNumber) {
      continue;
    }

    const quote = match[1].trim();

    if (
      String(text || "")
        .toLowerCase()
        .includes(quote.toLowerCase()) &&
      quote.length > longest.length
    ) {
      longest = quote;
    }
  }

  return longest;
}

function highlightedText(text, quote) {
  if (!quote) {
    return text;
  }

  const at = text
    .toLowerCase()
    .indexOf(quote.toLowerCase());

  if (at < 0) {
    return text;
  }

  return (
    <>
      {text.slice(0, at)}

      <mark className="evidence-highlight">
        {text.slice(at, at + quote.length)}
      </mark>

      {text.slice(at + quote.length)}
    </>
  );
}

function Answer({ message, openCitation }) {
  return (
    <div className="answer-markdown">
      <ReactMarkdown
        components={{
          a: ({ href, children }) => {
            const match =
              /^#source-(\d+)$/.exec(href || "");

            if (match) {
              return (
                <button
                  type="button"
                  className="citation"
                  onClick={() =>
                    openCitation(
                      message.id,
                      Number(match[1]) - 1
                    )
                  }
                  title={`Inspect evidence E${match[1]}`}
                >
                  {children}
                </button>
              );
            }

            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
              >
                {children}
              </a>
            );
          },
        }}
      >
        {readableAnswer(message.content)}
      </ReactMarkdown>
    </div>
  );
}

// ============================================================
// SPEECH EVIDENCE PANEL
// ============================================================

function EvidencePanel({
  message,
  selectedIndex,
  setSelectedIndex,
  onClose,
  copyText,
  copiedKey,
  citedOnly,
  setCitedOnly,
  partyFilter,
  setPartyFilter,
  yearFilter,
  setYearFilter,
}) {
  const evidence = Array.isArray(message?.evidence)
    ? message.evidence
    : [];

  const citations = citedIndices(message?.content);

  const parties = [
    ...new Set(
      evidence.map((e) => e.party || "Unrecorded")
    ),
  ].sort();

  const years = [
    ...new Set(
      evidence
        .map((e) =>
          String(e.date || "").slice(0, 4)
        )
        .filter(Boolean)
    ),
  ].sort();

  const visible = evidence
    .map((entry, index) => ({
      entry,
      index,
    }))
    .filter(
      ({ entry, index }) =>
        (!citedOnly || citations.has(index)) &&
        (
          partyFilter === "all" ||
          (entry.party || "Unrecorded") === partyFilter
        ) &&
        (
          yearFilter === "all" ||
          String(entry.date || "").startsWith(yearFilter)
        )
    );

  const reset = () => {
    setCitedOnly(false);
    setPartyFilter("all");
    setYearFilter("all");
    setSelectedIndex(null);
  };

  return (
    <aside
      className="evidence-panel"
      aria-label="Parliamentary speech evidence"
    >
      <div className="evidence-header">
        <div className="evidence-heading">
          <Database size={19} />

          <div>
            <strong>Source evidence</strong>
            <span>Parliamentary speech records</span>
          </div>
        </div>

        <button
          type="button"
          className="icon-button"
          onClick={onClose}
          aria-label="Close evidence panel"
        >
          <X size={18} />
        </button>
      </div>

      <div className="evidence-content">
        {!evidence.length ? (
          <div className="evidence-empty">
            <BookOpen size={27} />

            <p>
              Ask a question to inspect its supporting
              speeches.
            </p>
          </div>
        ) : (
          <>
            <div className="evidence-controls">
              <span>
                {evidence.length} passages ·{" "}
                {citations.size} cited
              </span>

              <label>
                <input
                  type="checkbox"
                  checked={citedOnly}
                  onChange={(e) => {
                    setCitedOnly(e.target.checked);
                    setSelectedIndex(null);
                  }}
                />

                Cited only
              </label>
            </div>

            <div className="evidence-filter-heading">
              <Filter size={13} />
              Filter evidence
            </div>

            <div className="evidence-filter-grid">
              <label>
                <span>Party</span>

                <select
                  value={partyFilter}
                  onChange={(e) => {
                    setPartyFilter(e.target.value);
                    setSelectedIndex(null);
                  }}
                >
                  <option value="all">
                    All parties
                  </option>

                  {parties.map((party) => (
                    <option key={party} value={party}>
                      {party}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                <span>Year</span>

                <select
                  value={yearFilter}
                  onChange={(e) => {
                    setYearFilter(e.target.value);
                    setSelectedIndex(null);
                  }}
                >
                  <option value="all">
                    All years
                  </option>

                  {years.map((year) => (
                    <option key={year} value={year}>
                      {year}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="source-result-count">
              Showing {visible.length} of{" "}
              {evidence.length} passages
            </div>

            {!visible.length && (
              <div className="source-no-results">
                No evidence matches these filters.

                <button
                  type="button"
                  onClick={reset}
                >
                  Reset filters
                </button>
              </div>
            )}

            {visible.map(({ entry, index }) => {
              const expanded =
                selectedIndex === index;

              const text = String(entry.text || "");

              const quote = matchingQuote(
                message?.content,
                index + 1,
                text
              );

              const key =
                `source-${message.id}-${index}`;

              return (
                <article
                  className={`source-card ${
                    expanded ? "selected" : ""
                  }`}
                  id={key}
                  key={key}
                >
                  <button
                    type="button"
                    className="source-card-trigger"
                    aria-expanded={expanded}
                    onClick={() =>
                      setSelectedIndex(
                        expanded ? null : index
                      )
                    }
                  >
                    <span className="source-card-top">
                      <span className="source-number">
                        E{index + 1}
                      </span>

                      {citations.has(index) && (
                        <span className="cited-tag">
                          <Check size={12} />
                          Cited
                        </span>
                      )}

                      {expanded ? (
                        <ChevronUp size={15} />
                      ) : (
                        <ChevronDown size={15} />
                      )}
                    </span>

                    <strong className="source-speaker">
                      {entry.speaker ||
                        "Unknown speaker"}
                    </strong>

                    <span className="source-meta">
                      {entry.party ||
                        "Party not recorded"}
                      {" · "}
                      {entry.date || "Undated"}
                    </span>

                    {entry.agenda && (
                      <span className="source-agenda">
                        {entry.agenda}
                      </span>
                    )}
                  </button>

                  {expanded ? (
                    <div className="expanded-source">
                      {quote && (
                        <div className="matched-excerpt">
                          <span>
                            Passage quoted in answer
                          </span>

                          <blockquote>
                            {quote}
                          </blockquote>
                        </div>
                      )}

                      <div className="source-text source-full-text">
                        {highlightedText(text, quote)}
                      </div>

                      <button
                        type="button"
                        className="copy-source-button"
                        onClick={() =>
                          copyText(
                            `E${index + 1} — ${
                              entry.speaker
                            } (${
                              entry.party
                            }, ${
                              entry.date
                            })\n${
                              entry.agenda || ""
                            }\n\n${text}`,
                            key
                          )
                        }
                      >
                        {copiedKey === key ? (
                          <Check size={14} />
                        ) : (
                          <Copy size={14} />
                        )}

                        {copiedKey === key
                          ? "Copied"
                          : "Copy speech"}
                      </button>
                    </div>
                  ) : (
                    <p className="source-text source-preview">
                      {text.slice(0, 165)}
                      {text.length > 165 ? "…" : ""}
                    </p>
                  )}
                </article>
              );
            })}

            <p className="evidence-disclaimer">
              Citations identify retrieved passages;
              historical interpretations still require
              source review.
            </p>
          </>
        )}
      </div>
    </aside>
  );
}

// ============================================================
// MAIN APPLICATION
// ============================================================

export default function App() {
  const [conversations, setConversations] =
    useState([]);

  const [activeId, setActiveId] =
    useState(null);

  const [ready, setReady] =
    useState(false);

  const [historyWarning, setHistoryWarning] =
    useState("");

  const [historySearch, setHistorySearch] =
    useState("");

  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);

  const [apiPath, setApiPath] = useState(() => {
    try {
      const saved =
        window.localStorage.getItem(
          "parliamentai-endpoint"
        );

      return saved && saved !== "/ask"
        ? saved
        : API_ROUTE;
    } catch {
      return API_ROUTE;
    }
  });

  const [settingsOpen, setSettingsOpen] =
    useState(false);

  const [connectionNote, setConnectionNote] =
    useState("");

  const [sourcesOpen, setSourcesOpen] =
    useState(true);

  const [menuOpen, setMenuOpen] =
    useState(false);

  const [selectedMessageId, setSelectedMessageId] =
    useState(null);

  const [selectedIndex, setSelectedIndex] =
    useState(null);

  const [citedOnly, setCitedOnly] =
    useState(false);

  const [partyFilter, setPartyFilter] =
    useState("all");

  const [yearFilter, setYearFilter] =
    useState("all");

  const [copiedKey, setCopiedKey] =
    useState("");

  const [exportOpen, setExportOpen] =
    useState(false);

  const [exportNotice, setExportNotice] =
    useState("");

  const controllerRef = useRef(null);
  const requestSeq = useRef(0);
  const bottomRef = useRef(null);

  // ==========================================================
  // RESTORE V3 CONVERSATION HISTORY
  // ==========================================================

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const stored = await loadHistory();

        if (cancelled) {
          return;
        }

        const restored = Array.isArray(
          stored?.conversations
        )
          ? stored.conversations
              .filter(
                (c) =>
                  c &&
                  typeof c.id === "string" &&
                  Array.isArray(c.messages)
              )
              .map((c) => ({
                ...c,

                messages: c.messages.map((m) =>
                  m.pending
                    ? {
                        ...m,
                        pending: false,
                        error:
                          "Request interrupted by a reload. Ask again.",
                      }
                    : m
                ),
              }))
          : [];

        const chats = restored.length
          ? restored
          : [newConversation()];

        setConversations(chats);

        setActiveId(
          chats.some(
            (c) => c.id === stored?.activeId
          )
            ? stored.activeId
            : chats[0].id
        );
      } catch {
        if (cancelled) {
          return;
        }

        const first = newConversation();

        setConversations([first]);
        setActiveId(first.id);

        setHistoryWarning(
          "Browser storage is unavailable. " +
          "Conversations may not survive a refresh."
        );
      } finally {
        if (!cancelled) {
          setReady(true);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // ==========================================================
  // AUTOMATIC PERSISTENCE
  // ==========================================================

  useEffect(() => {
    if (!ready) {
      return;
    }

    saveHistory({
      conversations,
      activeId,
    }).catch(() =>
      setHistoryWarning(
        "Could not save conversation history. " +
        "Check browser storage permissions."
      )
    );
  }, [conversations, activeId, ready]);

  useEffect(() => {
    try {
      localStorage.setItem(
        "parliamentai-endpoint",
        apiPath
      );
    } catch {
      // Local storage is optional.
    }
  }, [apiPath]);

  const activeChat = conversations.find(
    (c) => c.id === activeId
  );

  const messages = useMemo(() => activeChat?.messages ?? [], [activeChat]);

  const currentAnswer = useMemo(
    () =>
      messages.find(
        (m) =>
          m.id === selectedMessageId &&
          m.role === "assistant"
      ) ||
      [...messages]
        .reverse()
        .find(
          (m) =>
            m.role === "assistant" &&
            !m.pending
        ),
    [messages, selectedMessageId]
  );

  const historyItems = useMemo(
    () =>
      conversations
        .filter(
          (c) => c.messages?.length
        )
        .filter((c) =>
          `${
            c.title
          } ${
            c.messages
              .filter(
                (m) => m.role === "user"
              )
              .map(
                (m) => m.content
              )
              .join(" ")
          }`
            .toLowerCase()
            .includes(
              historySearch.trim().toLowerCase()
            )
        )
        .sort((a, b) =>
          String(b.updatedAt).localeCompare(
            String(a.updatedAt)
          )
        ),
    [conversations, historySearch]
  );

  // ==========================================================
  // SCROLLING AND CLEANUP
  // ==========================================================

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [conversations, activeId]);

  useEffect(
    () => () =>
      controllerRef.current?.controller.abort(),
    []
  );

  useEffect(() => {
    if (
      !sourcesOpen ||
      selectedIndex === null ||
      !currentAnswer
    ) {
      return;
    }

    const frame = requestAnimationFrame(() =>
      document
        .getElementById(
          `source-${currentAnswer.id}-${selectedIndex}`
        )
        ?.scrollIntoView({
          behavior: "smooth",
          block: "nearest",
        })
    );

    return () => cancelAnimationFrame(frame);
  }, [
    sourcesOpen,
    selectedIndex,
    currentAnswer,
  ]);

  // ==========================================================
  // RESET SOURCE VIEW
  // ==========================================================

  function resetSourceView() {
    setSelectedIndex(null);
    setSelectedMessageId(null);

    setPartyFilter("all");
    setYearFilter("all");
    setCitedOnly(false);
    setCopiedKey("");
  }

  // ==========================================================
  // CANCEL AN ACTIVE REQUEST
  // ==========================================================

  function cancelRequest() {
    const request = controllerRef.current;

    if (!request) {
      return;
    }

    requestSeq.current += 1;

    request.controller.abort();

    window.clearTimeout(
      request.timeout
    );

    controllerRef.current = null;
    setBusy(false);

    setConversations((old) =>
      old.map((c) =>
        c.id === request.chatId
          ? {
              ...c,

              messages: c.messages.map((m) =>
                m.id === request.answerId
                  ? {
                      ...m,
                      pending: false,
                      error:
                        "Request cancelled. You can ask again.",
                    }
                  : m
              ),
            }
          : c
      )
    );
  }

  // ==========================================================
  // NAVIGATE CONVERSATIONS
  // ==========================================================

  function selectChat(id) {
    if (id !== activeId) {
      cancelRequest();
      setActiveId(id);
      setDraft("");
      resetSourceView();
    }

    setMenuOpen(false);
    setExportOpen(false);
    setExportNotice("");
  }

  function startNewChat() {
    cancelRequest();

    if (
      !activeChat ||
      activeChat.messages.length
    ) {
      const chat = newConversation();

      setConversations((old) => [
        chat,
        ...old,
      ]);

      setActiveId(chat.id);
    }

    setDraft("");
    resetSourceView();
    setMenuOpen(false);
    setExportOpen(false);
    setExportNotice("");
  }

  function deleteChat(id) {
    if (
      !window.confirm(
        "Delete this conversation and its saved evidence?"
      )
    ) {
      return;
    }

    if (
      controllerRef.current?.chatId === id
    ) {
      cancelRequest();
    }

    const remaining = conversations.filter(
      (c) => c.id !== id
    );

    const next = remaining.length
      ? remaining
      : [newConversation()];

    setConversations(next);

    if (activeId === id) {
      setActiveId(next[0].id);
      resetSourceView();
    }

    setExportOpen(false);
    setExportNotice("");
  }

  // ==========================================================
  // CITATION NAVIGATION
  // ==========================================================

  function openCitation(messageId, index) {
    const entry = messages.find(
      (m) => m.id === messageId
    );

    if (
      !entry ||
      index < 0 ||
      index >=
        (entry.evidence || []).length
    ) {
      return;
    }

    setSelectedMessageId(messageId);
    setSelectedIndex(index);

    setCitedOnly(false);
    setPartyFilter("all");
    setYearFilter("all");
    setSourcesOpen(true);
  }

  // ==========================================================
  // COPY TEXT
  // ==========================================================

  async function copyText(text, key) {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedKey(key);
    } catch {
      window.alert(
        "Copy failed. Check your browser clipboard permissions."
      );
    }
  }

  // ==========================================================
  // EXPORT CURRENT OR ALL CONVERSATIONS
  // ==========================================================

  function exportData(kind) {
    try {
      const stamp = new Date()
        .toISOString()
        .replace(/[:.]/g, "-")
        .slice(0, 19);

      if (kind === "all-json") {
        const count = conversations.filter(
          (c) => c.messages?.length
        ).length;

        if (!count) {
          return;
        }

        downloadText(
          `parliamentai-all-conversations-${stamp}.json`,
          exportAllJSON(conversations),
          "application/json;charset=utf-8"
        );

        setExportNotice(
          `Started a JSON backup of ${count} conversations.`
        );
      } else {
        if (!activeChat?.messages.length) {
          return;
        }

        const base =
          `parliamentai-${
            safeFilename(activeChat.title)
          }-${stamp}`;

        if (kind === "markdown") {
          downloadText(
            `${base}.md`,
            exportConversationMarkdown(activeChat),
            "text/markdown;charset=utf-8"
          );
        } else if (kind === "json") {
          downloadText(
            `${base}.json`,
            exportConversationJSON(activeChat),
            "application/json;charset=utf-8"
          );
        }

        setExportNotice(
          `Started ${
            kind === "markdown"
              ? "Markdown"
              : "JSON"
          } export.`
        );
      }

      setExportOpen(false);
    } catch (error) {
      setExportNotice(
        `Export failed: ${error.message}`
      );
    }
  }

  // ==========================================================
  // DETECT GRAPHRAG API ROUTE
  // ==========================================================

  async function detectRoute() {
    setConnectionNote(
      "Checking FastAPI…"
    );

    try {
      const response = await fetch(
        "/backend/openapi.json"
      );

      if (!response.ok) {
        throw new Error(
          "FastAPI is not responding on port 8000."
        );
      }

      const spec = await response.json();

      const paths = Object.entries(
        spec.paths || {}
      )
        .filter(
          ([p, ops]) =>
            ops?.post &&
            !p.includes("{")
        )
        .map(([p]) => p);

      const found = paths.includes(API_ROUTE)
        ? API_ROUTE
        : paths.find((p) =>
            /graphrag/i.test(p)
          );

      if (!found) {
        throw new Error(
          "No GraphRAG POST route found."
        );
      }

      setApiPath(found);

      setConnectionNote(
        `Detected POST ${found}.`
      );
    } catch (error) {
      setConnectionNote(
        error.message
      );
    }
  }

  // ==========================================================
  // ASK GRAPHRAG
  // ==========================================================

  async function ask(rawQuestion = draft) {
    const question = String(
      rawQuestion || ""
    ).trim();

    if (
      !ready ||
      !question ||
      busy ||
      !activeId
    ) {
      return;
    }

    const chatId = activeId;
    const nonce = makeId();

    const answerId = `answer-${nonce}`;
    const now = new Date().toISOString();

    setConversations((old) =>
      old.map((chat) =>
        chat.id === chatId
          ? {
              ...chat,

              title: chat.messages.length
                ? chat.title
                : makeTitle(question),

              updatedAt: now,

              messages: [
                ...chat.messages,

                {
                  id: `user-${nonce}`,
                  role: "user",
                  content: question,
                },

                {
                  id: answerId,
                  role: "assistant",
                  content: "",
                  evidence: [],
                  pending: true,
                },
              ],
            }
          : chat
      )
    );

    setDraft("");
    setBusy(true);
    setMenuOpen(false);
    setExportOpen(false);

    resetSourceView();
    setSelectedMessageId(answerId);

    const seq = ++requestSeq.current;

    const controller = new AbortController();

    const timeout = window.setTimeout(
      () => controller.abort(),
      240000
    );

    controllerRef.current = {
      controller,
      timeout,
      chatId,
      answerId,
    };

    const replaceAnswer = (fields) => {
      if (seq !== requestSeq.current) {
        return;
      }

      setConversations((old) =>
        old.map((chat) =>
          chat.id === chatId
            ? {
                ...chat,

                updatedAt:
                  new Date().toISOString(),

                messages: chat.messages.map(
                  (m) =>
                    m.id === answerId
                      ? {
                          ...m,
                          ...fields,
                          pending: false,
                        }
                      : m
                ),
              }
            : chat
        )
      );
    };

    try {
      const route = apiPath.trim();

      if (
        !/^\/[A-Za-z0-9/_-]+$/.test(
          route
        )
      ) {
        throw new Error(
          "Invalid API route. Use /graphrag/ask."
        );
      }

      const response = await fetch(
        `/backend${route}`,
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json",
          },

          body: JSON.stringify({
            question,
          }),

          signal: controller.signal,
        }
      );

      const raw = await response.text();

      let result;

      try {
        result = JSON.parse(raw);
      } catch {
        throw new Error(
          `FastAPI returned non-JSON (HTTP ${response.status}).`
        );
      }

      if (!response.ok) {
        const detail =
          typeof result.detail === "string"
            ? result.detail
            : JSON.stringify(
                result.detail || ""
              );

        throw new Error(
          `FastAPI HTTP ${response.status}: ${detail}`
        );
      }

      if (
        typeof result.answer !== "string"
      ) {
        throw new Error(
          "The endpoint did not return an answer. Check /graphrag/ask."
        );
      }

      replaceAnswer({
        content: result.answer,

        evidence: Array.isArray(
          result.evidence
        )
          ? result.evidence
          : [],

        coverage:
          result.coverage || {},

        retrieval:
          result.retrieval || {},
      });
    } catch (error) {
      replaceAnswer({
        error:
          error.name === "AbortError"
            ? "Request cancelled or timed out."
            : error.message,
      });
    } finally {
      window.clearTimeout(timeout);

      if (seq === requestSeq.current) {
        controllerRef.current = null;
        setBusy(false);
      }
    }
  }

  const hasCurrent = Boolean(
    activeChat?.messages.length
  );

  const hasHistory = conversations.some(
    (c) => c.messages?.length
  );

  // ==========================================================
  // APPLICATION LAYOUT
  // ==========================================================

  return (
    <div
      className={`app-shell ${
        sourcesOpen
          ? ""
          : "without-sources"
      }`}
    >
      <style>{v4Styles}</style>

      {menuOpen && (
        <button
          type="button"
          className="mobile-shade"
          aria-label="Close navigation"
          onClick={() =>
            setMenuOpen(false)
          }
        />
      )}

      {/* SIDEBAR */}

      <nav
        className={`sidebar ${
          menuOpen
            ? "sidebar-open"
            : ""
        }`}
      >
        <div className="brand">
          <span className="brand-icon">
            <MessageSquareText size={21} />
          </span>

          <span>
            <strong>ParliamentAI</strong>
            <small>
              Brexit debate explorer
            </small>
          </span>
        </div>

        <button
          type="button"
          className="new-chat-button"
          onClick={startNewChat}
          disabled={!ready}
        >
          <Plus size={18} />
          New conversation
        </button>

        <div className="sidebar-subtitle">
          EXPLORE THE ARCHIVE
        </div>

        <div className="sidebar-suggestions">
          {QUESTIONS.map((q) => (
            <button
              type="button"
              key={q}
              onClick={() => ask(q)}
              disabled={busy || !ready}
            >
              <Search size={16} />
              <span>{q}</span>
              <ArrowUpRight size={14} />
            </button>
          ))}
        </div>

        {/* CONVERSATION HISTORY */}

        <div className="history-heading">
          <span>
            CONVERSATION HISTORY
          </span>

          <span>
            {historyItems.length}
          </span>
        </div>

        <div className="history-search">
          <Search size={14} />

          <input
            value={historySearch}
            onChange={(e) =>
              setHistorySearch(
                e.target.value
              )
            }
            placeholder="Search previous chats"
            aria-label="Search conversation history"
          />
        </div>

        <div className="history-list">
          {!historyItems.length ? (
            <p className="history-empty">
              {historySearch
                ? "No matching conversations."
                : "Your previous questions will appear here."}
            </p>
          ) : (
            historyItems.map((chat) => (
              <div
                key={chat.id}
                className={`history-row ${
                  chat.id === activeId
                    ? "active"
                    : ""
                }`}
              >
                <button
                  type="button"
                  className="history-open"
                  title={chat.title}
                  onClick={() =>
                    selectChat(chat.id)
                  }
                >
                  <MessageSquareText size={14} />

                  <span>
                    {chat.title}
                  </span>
                </button>

                <button
                  type="button"
                  className="history-delete"
                  title={`Delete ${chat.title}`}
                  aria-label={`Delete ${chat.title}`}
                  onClick={() =>
                    deleteChat(chat.id)
                  }
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))
          )}
        </div>

        <div className="sidebar-bottom">
          <div className="dataset-label">
            <Database size={15} />
            Speech archive · 2015–2019
          </div>

          <button
            type="button"
            className="settings-button"
            onClick={() => {
              setSettingsOpen(
                (open) => !open
              );

              setMenuOpen(false);
            }}
          >
            <Settings2 size={17} />
            API connection
          </button>
        </div>
      </nav>

      {/* MAIN CHAT */}

      <main className="chat-column">
        <header className="chat-header">
          <button
            type="button"
            className="icon-button mobile-menu"
            aria-label="Open navigation"
            onClick={() =>
              setMenuOpen(
                (open) => !open
              )
            }
          >
            <Menu size={20} />
          </button>

          <div className="chat-heading">
            <strong>
              Research assistant
            </strong>

            <small>
              Evidence-backed parliamentary research
            </small>
          </div>

          <span className="prototype-pill">
            Research prototype
          </span>

          {/* EXPORT MENU */}

          <div className="export-control">
            <button
              type="button"
              className="source-toggle export-toggle"
              onClick={() =>
                setExportOpen(
                  (open) => !open
                )
              }
              aria-expanded={exportOpen}
              aria-label="Open export options"
              disabled={!hasHistory}
            >
              <Download size={16} />
              Export
              <ChevronDown size={13} />
            </button>

            {exportOpen && (
              <div
                className="export-menu"
                role="menu"
              >
                <div className="export-menu-title">
                  Current conversation
                </div>

                <button
                  type="button"
                  role="menuitem"
                  disabled={!hasCurrent}
                  onClick={() =>
                    exportData("markdown")
                  }
                >
                  <FileText size={16} />

                  <span>
                    <strong>
                      Markdown (.md)
                    </strong>

                    <small>
                      Answer + full speech sources
                    </small>
                  </span>
                </button>

                <button
                  type="button"
                  role="menuitem"
                  disabled={!hasCurrent}
                  onClick={() =>
                    exportData("json")
                  }
                >
                  <FileJson size={16} />

                  <span>
                    <strong>
                      JSON (.json)
                    </strong>

                    <small>
                      Structured answers and evidence
                    </small>
                  </span>
                </button>

                <div className="export-menu-divider" />

                <button
                  type="button"
                  role="menuitem"
                  disabled={!hasHistory}
                  onClick={() =>
                    exportData("all-json")
                  }
                >
                  <Database size={16} />

                  <span>
                    <strong>
                      Back up all conversations
                    </strong>

                    <small>
                      JSON file including saved evidence
                    </small>
                  </span>
                </button>
              </div>
            )}
          </div>

          <button
            type="button"
            className="source-toggle"
            onClick={() =>
              setSourcesOpen(
                (open) => !open
              )
            }
          >
            <Database size={17} />
            Sources
          </button>
        </header>

        {/* NOTICES */}

        {historyWarning && (
          <div className="history-warning">
            <CircleAlert size={15} />
            {historyWarning}
          </div>
        )}

        {exportNotice && (
          <div
            className="export-notice"
            role="status"
          >
            <Check size={15} />

            <span>
              {exportNotice}
            </span>

            <button
              type="button"
              aria-label="Dismiss export notice"
              onClick={() =>
                setExportNotice("")
              }
            >
              <X size={13} />
            </button>
          </div>
        )}

        {/* API SETTINGS */}

        {settingsOpen && (
          <section className="connection-settings">
            <strong>
              FastAPI connection
            </strong>

            <p>
              The actual GraphRAG POST route is
              /graphrag/ask.
            </p>

            <div>
              <input
                value={apiPath}
                onChange={(e) =>
                  setApiPath(
                    e.target.value
                  )
                }
                aria-label="FastAPI endpoint"
              />

              <button
                type="button"
                onClick={detectRoute}
              >
                Detect
              </button>

              <button
                type="button"
                onClick={() =>
                  setSettingsOpen(false)
                }
              >
                Close
              </button>
            </div>

            {connectionNote && (
              <small>
                {connectionNote}
              </small>
            )}
          </section>
        )}

        {/* CONVERSATION */}

        <div className="conversation">
          {!ready ? (
            <div className="history-loading">
              <LoaderCircle
                size={17}
                className="spinner"
              />

              Loading saved conversations…
            </div>
          ) : !messages.length ? (
            <section className="welcome">
              <span className="welcome-icon">
                <BookOpen size={31} />
              </span>

              <span className="eyebrow">
                UK PARLIAMENT · 2015–2019
              </span>

              <h1>
                Explore the Brexit debate{" "}
                <em>
                  through the words of MPs.
                </em>
              </h1>

              <p>
                Ask about political positions,
                the withdrawal process, or changing
                parliamentary arguments. Inspect
                the original evidence behind each
                answer.
              </p>

              <div className="example-grid">
                {QUESTIONS.map((q) => (
                  <button
                    type="button"
                    key={q}
                    onClick={() =>
                      ask(q)
                    }
                  >
                    <span>
                      {q}
                    </span>

                    <ArrowUpRight size={17} />
                  </button>
                ))}
              </div>
            </section>
          ) : (
            <div className="messages">
              {messages.map((m) =>
                m.role === "user" ? (
                  <div
                    className="user-row"
                    key={m.id}
                  >
                    <div className="user-bubble">
                      {m.content}
                    </div>
                  </div>
                ) : (
                  <article
                    className="assistant-row"
                    key={m.id}
                  >
                    <div className="assistant-icon">
                      <BookOpen size={17} />
                    </div>

                    <div className="assistant-body">
                      <div className="assistant-name">
                        ParliamentAI{" "}

                        <small>
                          · from retrieved parliamentary
                          speeches
                        </small>
                      </div>

                      {m.pending ? (
                        <div className="loading">
                          <LoaderCircle
                            size={17}
                            className="spinner"
                          />

                          Searching the archive…
                        </div>
                      ) : m.error ? (
                        <div className="error-message">
                          <CircleAlert size={18} />
                          {m.error}
                        </div>
                      ) : (
                        <>
                          <Answer
                            message={m}
                            openCitation={openCitation}
                          />

                          <div className="answer-actions">
                            <button
                              type="button"
                              className="view-sources"
                              onClick={() => {
                                setSelectedMessageId(m.id);
                                setSelectedIndex(null);
                                setCitedOnly(false);
                                setPartyFilter("all");
                                setYearFilter("all");
                                setSourcesOpen(true);
                              }}
                            >
                              <Database size={15} />

                              {(m.evidence || []).length}{" "}
                              evidence passages

                              <ArrowUpRight size={14} />
                            </button>

                            <button
                              type="button"
                              className="copy-answer-button"
                              onClick={() =>
                                copyText(
                                  m.content,
                                  m.id
                                )
                              }
                            >
                              {copiedKey === m.id ? (
                                <Check size={14} />
                              ) : (
                                <Copy size={14} />
                              )}

                              {copiedKey === m.id
                                ? "Copied"
                                : "Copy answer"}
                            </button>
                          </div>

                          {m.coverage?.coverage_ok === false && (
                            <p className="coverage-warning">
                              The backend reported
                              incomplete evidence coverage.
                            </p>
                          )}
                        </>
                      )}
                    </div>
                  </article>
                )
              )}

              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* MESSAGE INPUT */}

        <form
          className="composer-area"
          onSubmit={(e) => {
            e.preventDefault();
            ask();
          }}
        >
          <div className="composer">
            <textarea
              value={draft}
              onChange={(e) =>
                setDraft(e.target.value)
              }
              rows={2}
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" &&
                  !e.shiftKey &&
                  !e.nativeEvent.isComposing
                ) {
                  e.preventDefault();
                  ask();
                }
              }}
              disabled={busy || !ready}
              placeholder="Ask a question about Parliament and Brexit…"
              aria-label="Your question"
            />

            {busy ? (
              <button
                type="button"
                className="send-button"
                title="Cancel request"
                onClick={cancelRequest}
              >
                <X size={18} />
              </button>
            ) : (
              <button
                type="submit"
                className="send-button"
                disabled={
                  !ready ||
                  !draft.trim()
                }
                aria-label="Send question"
              >
                <Send size={18} />
              </button>
            )}
          </div>

          <small>
            Saved locally in this browser ·
            Export research anytime ·
            Verify source context
          </small>
        </form>
      </main>

      {/* EVIDENCE PANEL */}

      {sourcesOpen && (
        <EvidencePanel
          message={currentAnswer}
          selectedIndex={selectedIndex}
          setSelectedIndex={setSelectedIndex}
          onClose={() =>
            setSourcesOpen(false)
          }
          copyText={copyText}
          copiedKey={copiedKey}
          citedOnly={citedOnly}
          setCitedOnly={setCitedOnly}
          partyFilter={partyFilter}
          setPartyFilter={setPartyFilter}
          yearFilter={yearFilter}
          setYearFilter={setYearFilter}
        />
      )}
    </div>
  );
}

// ============================================================
// STYLES — ADDITIONS TO EXISTING index.css
// ============================================================

const v4Styles = `
.sidebar-suggestions {
  flex: 0 0 auto;
  max-height: 190px;
  overflow-y: auto;
}

.history-heading {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 24px 10px 10px;
  color: #83a9a7;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 1.25px;
}

.history-heading span:last-child {
  color: #d4e2df;
  letter-spacing: 0;
  font-size: 11px;
}

.history-search {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 9px 10px;
  margin-bottom: 10px;
  background: #1c4148;
  color: #8fb8b0;
  border: 1px solid #31525a;
  border-radius: 9px;
}

.history-search input {
  background: transparent;
  border: 0;
  outline: 0;
  color: #e9f3ee;
  font-size: 11px;
  min-width: 0;
  flex: 1;
}

.history-search input::placeholder {
  color: #86a9aa;
}

.history-list {
  min-height: 0;
  flex: 1;
  overflow-y: auto;
  padding-right: 3px;
}

.history-row {
  display: flex;
  align-items: center;
  border-radius: 9px;
  margin-bottom: 4px;
}

.history-row:hover {
  background: #21444b;
}

.history-row.active {
  background: #2a5355;
}

.history-open {
  min-width: 0;
  flex: 1;
  display: flex;
  align-items: center;
  gap: 9px;
  background: transparent;
  border: 0;
  color: #dceae5;
  padding: 11px 9px;
  text-align: left;
  font-size: 11px;
}

.history-open span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.history-open svg {
  flex-shrink: 0;
  color: #8ebaae;
}

.history-delete {
  flex-shrink: 0;
  border: 0;
  background: transparent;
  color: #9bb9b1;
  padding: 8px;
  border-radius: 6px;
  opacity: .65;
}

.history-delete:hover {
  color: #ffb9aa;
  opacity: 1;
}

.history-empty {
  color: #86a9aa;
  padding: 0 11px;
  font-size: 11px;
  line-height: 1.6;
}

.history-warning {
  display: flex;
  gap: 8px;
  align-items: center;
  background: #fff4dd;
  color: #806423;
  font-size: 12px;
  padding: 10px 22px;
}

.history-loading {
  display: flex;
  justify-content: center;
  gap: 9px;
  align-items: center;
  margin: 80px auto;
  font-size: 13px;
  color: #53826d;
}

.answer-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 18px;
  margin-top: 7px;
}

.copy-answer-button,
.copy-source-button {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  border: 0;
  background: transparent;
  color: #3a7966;
  font-size: 12px;
  font-weight: 700;
  padding: 8px 0;
}

.copy-answer-button:hover,
.copy-source-button:hover {
  text-decoration: underline;
}

.evidence-filter-heading {
  display: flex;
  align-items: center;
  gap: 6px;
  color: #637d73;
  font-size: 11px;
  font-weight: 750;
  margin: 15px 0 8px;
}

.evidence-filter-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-bottom: 12px;
}

.evidence-filter-grid label {
  min-width: 0;
}

.evidence-filter-grid label span {
  display: block;
  color: #789488;
  font-size: 10px;
  margin-bottom: 4px;
}

.evidence-filter-grid select {
  width: 100%;
  min-width: 0;
  padding: 8px 5px;
  border: 1px solid #dbe8df;
  border-radius: 8px;
  background: #fff;
  color: #325d4c;
  font-size: 11px;
}

.source-result-count {
  font-size: 10px;
  color: #82988e;
  margin-bottom: 13px;
}

.source-no-results {
  font-size: 12px;
  line-height: 1.7;
  color: #6a8578;
  background: #f4f8f5;
  border-radius: 10px;
  padding: 15px;
}

.source-no-results button {
  display: block;
  margin-top: 10px;
  background: transparent;
  border: none;
  color: #2b7856;
  padding: 0;
  text-decoration: underline;
}

.source-card-trigger {
  width: 100%;
  text-align: left;
  display: block;
  color: inherit;
  background: transparent;
  border: 0;
  padding: 0;
}

.source-preview {
  margin-bottom: 0;
}

.source-full-text {
  margin: 11px 0 6px;
  max-height: none;
}

.matched-excerpt {
  background: #ecf7ee;
  border-left: 3px solid #6cab88;
  border-radius: 5px;
  padding: 10px;
  margin-top: 13px;
}

.matched-excerpt > span {
  font-size: 10px;
  font-weight: 750;
  color: #367b57;
}

.matched-excerpt blockquote {
  margin: 6px 0 0;
  font-size: 11px;
  line-height: 1.7;
  color: #3e6e59;
  overflow-wrap: anywhere;
}

.evidence-highlight {
  color: inherit;
  background: #d5edb8;
  padding: 1px 2px;
  border-radius: 3px;
}

.expanded-source .copy-source-button {
  margin-top: 10px;
}

.answer-markdown p {
  margin-bottom: 18px;
}

.answer-markdown p + p {
  margin-top: 12px;
}

/* V4 export controls */

.export-control {
  position: relative;
  flex-shrink: 0;
}

.export-toggle {
  display: inline-flex;
  align-items: center;
  gap: 7px;
}

.export-menu {
  position: absolute;
  z-index: 30;
  right: 0;
  top: calc(100% + 10px);
  width: 295px;
  border: 1px solid #d6e5df;
  border-radius: 13px;
  background: white;
  box-shadow: 0 12px 30px #183f3524;
  padding: 7px;
}

.export-menu-title {
  font-size: 10px;
  font-weight: 800;
  letter-spacing: .5px;
  color: #7d958a;
  padding: 9px 10px 5px;
  text-transform: uppercase;
}

.export-menu > button {
  display: flex;
  align-items: flex-start;
  width: 100%;
  gap: 10px;
  text-align: left;
  background: transparent;
  border: 0;
  border-radius: 8px;
  padding: 11px 10px;
  color: #28634f;
}

.export-menu > button:hover:not(:disabled) {
  background: #f0f8f2;
}

.export-menu > button > svg {
  flex-shrink: 0;
  margin-top: 3px;
}

.export-menu strong,
.export-menu small {
  display: block;
}

.export-menu strong {
  font-size: 12px;
  color: #234d3d;
}

.export-menu small {
  font-size: 10px;
  color: #80948b;
  margin-top: 3px;
  line-height: 1.4;
}

.export-menu-divider {
  height: 1px;
  background: #e5efe8;
  margin: 4px 6px;
}

.export-notice {
  display: flex;
  align-items: center;
  gap: 8px;
  background: #eaf6ed;
  color: #326e54;
  padding: 9px 24px;
  font-size: 12px;
}

.export-notice span {
  flex: 1;
}

.export-notice button {
  background: transparent;
  border: 0;
  color: inherit;
  padding: 3px;
}

@media (max-height: 750px) {
  .sidebar-suggestions {
    max-height: 125px;
  }

  .sidebar-subtitle {
    margin-top: 20px;
  }
}

@media (max-width: 680px) {
  .answer-actions {
    gap: 9px;
  }

  .history-list {
    min-height: 100px;
  }

  .export-menu {
    right: -65px;
    width: min(295px, 86vw);
  }

  .export-toggle {
    padding: 8px;
    font-size: 11px;
  }

  .export-toggle svg:last-child {
    display: none;
  }
}
`;
