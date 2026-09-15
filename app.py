"""Streamlit interface for Fineprint.

This module is presentation only. Every decision about retrieval, refusal and
generation lives in rag/, so the same behaviour is available from the CLI and
from the evaluation harness.
"""

from __future__ import annotations

import html
import json
import re
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import config
from rag import db, ingest, loaders, pipeline
from rag import threshold as auto_threshold
from rag.audit import audit_answer
from rag.embeddings import Embedder
from rag.llm import ChatModel
from rag.retrieval import Index, Mode

CONV_PATH = config.DATA_DIR / "conversations.json"

st.set_page_config(
    page_title="Fineprint",
    page_icon="F",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Questions that each exercise a different path: a clean lookup, a colliding
# deadline, a refusal because the answer is absent, and a refusal because the
# question asks for advice.
EXAMPLE_QUESTIONS = [
    "How many days do I have to send a proof of loss after a flood loss?",
    "Within how many hours must an employer report a workplace fatality?",
    "By when can a buyer cancel a door-to-door sale under the Cooling-Off Rule?",
    "Under the condominium building policy, when is a coinsurance penalty imposed?",
    "What is the federal minimum wage?",
    "Should I file a flood insurance claim for this damage?",
]


def _library_is_sample_corpus_only(conn) -> bool:
    """Whether every document in the library also belongs to the sample corpus.

    EXAMPLE_QUESTIONS is written for that corpus specifically. Once someone
    adds their own document, those questions stop being a helpful demo and
    start being noise about paperwork that is not theirs, so they are shown
    only while the library has not diverged from the sample corpus.
    """
    sample_filenames = {path.name for path in loaders.discover_documents(config.EVAL_CORPUS_DIR)}
    library_filenames = {entry["filename"] for entry in db.list_documents(conn)}
    return bool(library_filenames) and library_filenames <= sample_filenames


STYLE = """
<style>
:root {
    --fp-accent: #10b981;
    --fp-line: rgba(128,128,128,0.20);
    --fp-line-hi: rgba(128,128,128,0.42);
    --fp-wash: rgba(128,128,128,0.055);
    --fp-compose: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' viewBox='0 0 24 24'><path d='M12 20h9'/><path d='M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z'/></svg>");
    --fp-search: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' viewBox='0 0 24 24'><circle cx='11' cy='11' r='7'/><path d='m21 21-4.3-4.3'/></svg>");
    /* a document with a folded corner and two lines of text: the sources
       trigger opens the passages that answered the question */
    --fp-doc: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' viewBox='0 0 24 24'><path d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z'/><path d='M14 2v6h6'/><path d='M8 13h8'/><path d='M8 17h5'/></svg>");
}

/* empty-state greeting in the main column, in place of a big page title */
.fp-hero { text-align: center; margin: 3.2rem 0 2.2rem; }
.fp-hero b { display: block; font-size: 1.95rem; font-weight: 650;
    letter-spacing: -0.015em; }
/* the headline b is the block; a b inside the sub-line is just emphasis */
.fp-hero span b { display: inline; font-size: inherit; font-weight: 700;
    opacity: 1; }
.fp-hero span { display: block; margin-top: 0.65rem; font-size: 1.02rem;
    opacity: 0.55; line-height: 1.55; }

.block-container { max-width: 780px; padding-top: 2.2rem; }
/* the cache_resource spinner ("Starting Fineprint and loading the local
   models...") can paint before the block-container's own top padding has
   taken effect, sitting flush against the very top edge for that first
   moment. Its own margin does not depend on that timing. */
div[data-testid="stSpinner"] { margin-top: 1.5rem; }
h1 { font-weight: 750; letter-spacing: -0.025em; margin-bottom: 0.1rem;
    font-size: 2rem; }
.fp-tagline { opacity: 0.62; font-size: 0.92rem; line-height: 1.5;
    margin: 0.1rem 0 1.8rem; }
.fp-label { text-transform: uppercase; letter-spacing: 0.12em;
    font-size: 0.82rem; opacity: 0.5; font-weight: 600; margin: 1.4rem 0 0.9rem; }
/* the centred "Try one" label above the example grid reads larger */
.st-key-emptydock .fp-label { font-size: 0.9rem; text-align: center;
    opacity: 0.55; margin: 0.4rem 0 1rem; }

/* --- suggestion cards (first screen) --- */
[class*="st-key-sugg_"] button {
    width: 100%; height: auto !important; min-height: 4.6rem;
    padding: 1.05rem 1.2rem !important; border-radius: 14px;
    font-weight: 450; font-size: 0.96rem; line-height: 1.5;
    border: 1px solid var(--fp-line); background: var(--fp-wash);
    display: flex; align-items: center;
    transition: transform .14s ease, border-color .14s ease, background .14s ease;
}
/* more air between the example rows and columns */
.st-key-emptydock [data-testid="stHorizontalBlock"] { gap: 0.9rem; }
.st-key-emptydock [data-testid="stVerticalBlock"] { gap: 0.9rem; }
[class*="st-key-sugg_"] button p {
    white-space: normal !important; text-align: left; margin: 0;
}
[class*="st-key-sugg_"] button:hover {
    transform: translateY(-2px);
    border-color: rgba(16,185,129,0.55);
    background: rgba(16,185,129,0.09);
}

/* --- faded "more to try" row (after first answer) --- */
[class*="st-key-hint_"] button {
    width: 100%; height: auto !important; min-height: 2.6rem;
    padding: 0.45rem 0.75rem !important; border-radius: 11px;
    font-size: 0.78rem; line-height: 1.35; font-weight: 400;
    border: 1px dashed var(--fp-line); background: transparent;
    opacity: 0.5; transition: opacity .14s ease, border-color .14s ease;
}
[class*="st-key-hint_"] button p { white-space: normal !important; margin: 0; }
[class*="st-key-hint_"] button:hover { opacity: 1; border-color: var(--fp-line-hi); }

/* --- chat bubbles --- */
div[data-testid="stChatMessage"] {
    padding: 0.55rem 0.2rem; border-radius: 16px; background: transparent;
}
/* avatars: a monogram tile for you (same identity system as the account chip
   in the sidebar) and a quotation mark for Fineprint, not a robot face or a
   generic sparkle. The glyph is absolutely centred so it never depends on
   Streamlit's inner icon span, which was what pulled the old icons off-centre. */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    position: relative; overflow: hidden;
    border: none !important; border-radius: 9px !important;
    width: 30px !important; height: 30px !important; flex: none !important;
    font-size: 0 !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.07);
}
[data-testid="stChatMessageAvatarUser"] > *,
[data-testid="stChatMessageAvatarAssistant"] > * { display: none !important; }
[data-testid="stChatMessageAvatarUser"]::after,
[data-testid="stChatMessageAvatarAssistant"]::after {
    content: ""; position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 650; line-height: 1;
    font-family: "Source Sans Pro", system-ui, sans-serif;
}
/* you: a monogram, same as the sidebar account chip */
[data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(140deg,#414d61 0%,#2b3444 55%,#333c4d 100%) !important;
    color: #d6dbe2 !important;
}
[data-testid="stChatMessageAvatarUser"]::after { content: "U"; }
/* Fineprint: a quotation mark, because every answer is a quoted clause */
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(140deg,rgba(16,185,129,0.32),rgba(16,185,129,0.14)) !important;
    color: #34d8a6 !important;
}
[data-testid="stChatMessageAvatarAssistant"]::after {
    background: currentColor;
    -webkit-mask: var(--fp-quote) center / 15px 15px no-repeat;
    mask: var(--fp-quote) center / 15px 15px no-repeat;
    --fp-quote: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='black'><path d='M4 14.5C4 10 6.6 7 10.5 6.2l.5 2C8.6 8.8 7.4 10 7.1 11.6H10V18H4zM14 14.5C14 10 16.6 7 20.5 6.2l.5 2c-2.4.6-3.6 1.8-3.9 3.4H20V18h-6z'/></svg>");
}
/* user turn: a bubble pinned to the right, like a messaging app. flex-end is
   set explicitly because Streamlit's own rule otherwise spaces the avatar and
   the bubble apart instead of packing them together. */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    /* row-reverse flips which edge "start"/"end" mean: flex-start is what
       packs the avatar and bubble against the true right edge here. */
    flex-direction: row-reverse; justify-content: flex-start !important;
    gap: 0.5rem; background: transparent;
}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"] {
    max-width: 72%; flex: 0 1 auto; margin: 0 !important;
    padding: 0.45rem 0.95rem;
    background: var(--fp-wash);
    border: 1px solid var(--fp-line-hi);
    border-radius: 16px;
}
/* assistant turn stays on the left, held to a column so it reads as a reply */
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
    [data-testid="stChatMessageContent"] {
    max-width: 90%;
}
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    font-size: 1.045rem; line-height: 1.66;
}

/* outcome notes (info / warning / error): quiet cards, not loud banners, so
   the answer text above stays the thing the eye lands on first */
div[data-testid="stAlertContainer"] {
    padding: 0.6rem 0.9rem !important; border-radius: 12px;
    font-size: 0.85rem !important; line-height: 1.5;
    background: var(--fp-wash) !important;
    border: 1px solid var(--fp-line);
}
div[data-testid="stAlertContainer"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.85rem !important; opacity: 0.85;
}
div[data-testid="stAlertContentWarning"] {
    border-left: 2px solid rgba(217,164,6,0.55); padding-left: 0.7rem;
}
div[data-testid="stAlertContentInfo"] {
    border-left: 2px solid rgba(96,165,250,0.55); padding-left: 0.7rem;
}
div[data-testid="stAlertContentError"] {
    border-left: 2px solid rgba(229,104,107,0.55); padding-left: 0.7rem;
}
div[data-testid="stAlertContainer"] [data-testid="stIconMaterial"] {
    font-size: 1rem !important; opacity: 0.75;
}

/* the eyebrow label above an answer */
.fp-eyebrow { text-transform: uppercase; letter-spacing: 0.09em;
    font-size: 0.82rem; font-weight: 700; margin: 0.15rem 0 0.5rem;
    color: var(--fp-accent); opacity: 0.85; }

/* --- popover panels (source chips, chat/document row menus): quieter than
   the answer, quoted with a left rule around their expander content --- */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px; border-color: var(--fp-line) !important;
    background: var(--fp-wash);
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stExpander"] summary p {
    font-size: 0.8rem !important; opacity: 0.82;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stExpanderDetails"] {
    border-left: 2px solid rgba(16,185,129,0.40);
    padding-left: 0.9rem;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stExpanderDetails"]
    [data-testid="stMarkdownContainer"] p {
    font-size: 0.85rem !important; line-height: 1.58; opacity: 0.82;
}
.fp-srcmeta { font-size: 0.73rem; opacity: 0.5; margin-top: 0.45rem;
    letter-spacing: 0.01em; }

/* --- sources trigger: opens one popover that holds both the passages and,
   nested at the bottom, "how this answer was built" - one floating panel,
   not two that could both end up open and overlap. --- */
[class*="st-key-srcrow_"] { margin: 0.6rem 0 0.1rem; }
[class*="st-key-srcpop_"] button {
    border-radius: 999px !important; border: 1px solid var(--fp-line-hi) !important;
    background: var(--fp-wash) !important; color: inherit !important;
    font-size: 0.8rem !important; font-weight: 600; letter-spacing: 0.01em;
    padding: 0.34rem 0.9rem 0.34rem 0.7rem !important;
    display: inline-flex !important; align-items: center; gap: 0.45rem;
    white-space: nowrap;
}
[class*="st-key-srcpop_"] button::before {
    content: ""; width: 14px; height: 14px; flex: none;
    background: currentColor; opacity: 0.8;
    -webkit-mask: var(--fp-doc) center / contain no-repeat;
    mask: var(--fp-doc) center / contain no-repeat;
}
[class*="st-key-srcpop_"] button:hover {
    background: rgba(16,185,129,0.12) !important;
    border-color: rgba(16,185,129,0.5) !important;
}
/* the dropdown panel: each source is its own expander, opened and closed
   independently instead of three panels all shown at once */
[class*="st-key-srcpop_"] [data-testid="stPopoverBody"] { min-width: 320px; }
[class*="st-key-srcpop_"] [data-testid="stExpander"] {
    border: none !important; background: transparent !important;
    border-bottom: 1px solid var(--fp-line) !important; border-radius: 0 !important;
}
[class*="st-key-srcpop_"] [data-testid="stExpander"]:last-of-type {
    border-bottom: none !important;
}
[class*="st-key-srcpop_"] [data-testid="stExpander"] summary {
    padding: 0.5rem 0.1rem !important;
}
[class*="st-key-srcpop_"] [data-testid="stExpander"] summary p {
    font-size: 0.82rem !important; font-weight: 550;
}
[class*="st-key-srcpop_"] [data-testid="stExpanderDetails"] {
    border-left: 2px solid rgba(16,185,129,0.40);
    padding: 0 0 0.7rem 0.8rem; margin-left: 0.1rem;
}

/* --- metrics strip under an answer --- */
.fp-metrics { font-size: 0.77rem; opacity: 0.5; margin-top: 0.8rem;
    line-height: 1.5; }
.fp-metrics b { font-weight: 650; opacity: 0.8; }

/* --- animated "thinking" line --- */
.fp-thinking { display: flex; align-items: center; gap: 0.55rem;
    font-size: 0.82rem; opacity: 0.8; margin: 0.15rem 0; }
.fp-thinking .d { width: 6px; height: 6px; border-radius: 50%;
    background: var(--fp-accent); animation: fp-bounce 1.2s infinite ease-in-out; }
.fp-thinking .d:nth-child(2) { animation-delay: 0.15s; }
.fp-thinking .d:nth-child(3) { animation-delay: 0.30s; }
@keyframes fp-bounce {
    0%, 80%, 100% { transform: scale(0.5); opacity: 0.3; }
    40% { transform: scale(1); opacity: 1; }
}

/* --- sidebar --- */
div[data-testid="stSidebar"] { border-right: 1px solid var(--fp-line); }
/* make the sidebar shell the containing block for the pinned foot + account
   chip (both position:fixed below), so they take the sidebar's real width
   even when it is resized, and still anchor to the viewport bottom. */
[data-testid="stSidebarContent"] { contain: layout !important; }
/* library + settings rows: borderless, like chatbot nav items */
div[data-testid="stSidebar"] div[data-testid="stExpander"] {
    border: none; border-radius: 8px; margin-bottom: 0.1rem;
    transition: background .14s ease;
}
div[data-testid="stSidebar"] div[data-testid="stExpander"] summary {
    padding: 0.3rem 0.5rem; font-size: 0.85rem;
}
div[data-testid="stSidebar"] div[data-testid="stExpander"]:hover {
    background: var(--fp-wash);
}

/* pin the account chip to the bottom of the sidebar: a flex column from the
   scroll container down to the vertical block, then margin-top:auto on the
   last element container (the chip). The header is left untouched. */
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"],
[data-testid="stSidebarUserContent"] > div,
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {
    display: flex; flex-direction: column; flex: 1 1 auto; min-height: 0;
}
/* belt and suspenders: the chat list caps and scrolls itself, but if the
   header area alone is ever taller than the window, let the whole sidebar
   scroll too rather than silently clipping content with no way to reach it. */
[data-testid="stSidebarContent"] { overflow-y: auto !important; }
/* library + settings: pinned to the sidebar bottom, just above the account
   chip. Fixed (like the chip) so it does not depend on Streamlit's flex chain
   filling the sidebar height; it scrolls internally if an expander is opened. */
.st-key-sidebarfoot {
    position: fixed; left: 0; right: 0; bottom: 3.7rem;
    box-sizing: border-box; padding: 0.5rem 1.25rem 0.45rem;
    background: #161a23;
    max-height: calc(100vh - 5.2rem); overflow-y: auto; z-index: 89;
}
/* keep the conversation list clear of the pinned block */
[data-testid="stSidebarUserContent"] { padding-bottom: 11rem; }

/* the conversation list scroll box: no border, blends in */
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"],
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] > div {
    border: none !important; box-shadow: none !important;
    background: transparent !important; padding: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    margin: 0.1rem 0 0.4rem;
}
.st-key-convsearch input { font-size: 0.85rem; }
div[data-testid="stSidebar"] .fp-label { margin: 0.7rem 0 0.35rem; }

/* the chat list scrolls on its own, capped well above the pinned foot, so a
   long history never gets trapped under it with no way to reach the rest */
.st-key-convlist {
    /* generous reservation for the header above and the pinned foot below
       (which can itself grow a little with search open) so the list's own
       scroll area never runs under the fixed foot, where clicks and wheel
       scroll would be swallowed by it instead of reaching the list */
    max-height: calc(100vh - 33rem); min-height: 6rem;
    overflow-y: auto; overflow-x: hidden;
    padding-right: 0.2rem;
}

/* conversation rows: the whole row (title + the 3-dot menu) sits on one faint
   card; the current row is a touch brighter with a hairline border */
[class*="st-key-convrow_"] [data-testid="stHorizontalBlock"] {
    gap: 0.1rem; align-items: center;
    background: rgba(255,255,255,0.035);
    border: 1px solid transparent; border-radius: 8px;
    padding-right: 0.15rem;
}
[class*="st-key-convrow_"]:hover [data-testid="stHorizontalBlock"] {
    background: var(--fp-wash);
}
[class*="st-key-convrow_"]:has(a.fp-convlink.active) [data-testid="stHorizontalBlock"] {
    background: rgba(255,255,255,0.10);
    border-color: var(--fp-line-hi);
}
/* a real <a href="?chat=..."> rather than a button: a normal left click still
   swaps chats, but right-click now offers the browser's own "open in new
   tab" / "copy link", which no st.button can ever do. */
a.fp-convlink {
    display: flex; align-items: center; width: 100%;
    padding: 0.42rem 0.6rem; border-radius: 8px; box-sizing: border-box;
    font-weight: 450; font-size: 0.85rem; line-height: 1.3;
    color: inherit !important; text-decoration: none !important;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
a.fp-convlink.active { font-weight: 550; }
a.fp-convlink:hover { text-decoration: none !important; }
[class*="st-key-convmenu_"] button {
    border: none !important; background: transparent !important;
    color: inherit !important; opacity: 0.3;
    min-width: 0 !important; padding: 0.3rem 0 !important;
    font-size: 1.05rem !important; line-height: 1;
}
[class*="st-key-convrow_"]:hover [class*="st-key-convmenu_"] button { opacity: 0.7; }
[class*="st-key-convmenu_"] button:hover { opacity: 1; background: var(--fp-wash) !important; }
[class*="st-key-convmenu_"] button svg,
[class*="st-key-convmenu_"] button [data-testid="stIconMaterial"] {
    display: none !important;
}
[class*="st-key-rename_"] input { font-size: 0.85rem; }
[class*="st-key-dodelete_"] button { color: #e5686b !important; }

/* document row menu: same quiet "⋮" as a chat row's menu */
[class*="st-key-docmenu_"] button {
    border: none !important; background: transparent !important;
    color: inherit !important; opacity: 0.45;
    min-width: 0 !important; padding: 0.3rem 0 !important;
    font-size: 1.05rem !important; line-height: 1;
}
[class*="st-key-docmenu_"] button:hover { opacity: 1; background: var(--fp-wash) !important; }
[class*="st-key-docmenu_"] button svg,
[class*="st-key-docmenu_"] button [data-testid="stIconMaterial"] {
    display: none !important;
}
[class*="st-key-dodeletedoc_"] button { color: #e5686b !important; }

/* the collapsed document list: tighter rows */
.fp-doc-sub { font-size: 0.72rem; opacity: 0.5; }
div[data-testid="stSidebar"] [data-testid="stExpanderDetails"] p {
    margin-bottom: 0.5rem;
}

/* account chip: minimal, like ChatGPT's, fixed to the sidebar bottom. It is a
   fixed child of the sidebar, so it rides the sidebar's slide-out when the
   sidebar is collapsed. 300px is Streamlit's default expanded width. */
.fp-user { position: fixed; left: 0; right: 0; bottom: 0.55rem;
    box-sizing: border-box; padding: 0.5rem 1.25rem;
    display: flex; align-items: center; gap: 0.6rem;
    font-size: 0.82rem; line-height: 1.25; cursor: default;
    background: #161a23; border-top: 1px solid var(--fp-line);
    transition: background .14s ease; z-index: 90; }
.fp-user:hover { background: #1b2130; }
.fp-user-dot { width: 30px; height: 30px; border-radius: 9px; flex: none;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.8rem; font-weight: 650; letter-spacing: 0.02em;
    color: #eceef2;
    background: linear-gradient(140deg, #414d61 0%, #2b3444 55%, #333c4d 100%);
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.07); }
.fp-user .sub { opacity: 0.5; font-size: 0.72rem; }

/* "New chat" + search: understated nav rows with an icon, chatbot style */
[class*="st-key-newchat"] button,
[class*="st-key-searchbtn"] button {
    border-radius: 8px !important; border: none !important;
    background: transparent !important; color: inherit !important;
    font-weight: 500; justify-content: flex-start !important;
    gap: 0.55rem; padding: 0.45rem 0.55rem !important; font-size: 0.88rem;
    transition: background .14s ease;
}
[class*="st-key-newchat"] button {
    margin: 0.2rem 0 0.3rem;
    border: 1px solid var(--fp-line-hi) !important;
    background: var(--fp-wash) !important;
}
[class*="st-key-newchat"] button::before,
[class*="st-key-searchbtn"] button::before {
    content: ""; width: 16px; height: 16px; flex: none;
    background: currentColor; opacity: 0.7;
    -webkit-mask: var(--fp-compose) center / contain no-repeat;
    mask: var(--fp-compose) center / contain no-repeat;
}
[class*="st-key-searchbtn"] button::before {
    -webkit-mask: var(--fp-search) center / contain no-repeat;
    mask: var(--fp-search) center / contain no-repeat;
}
[class*="st-key-newchat"] button:hover,
[class*="st-key-searchbtn"] button:hover { background: var(--fp-wash) !important; }
/* hide the popover's default trailing caret icon */
[class*="st-key-searchbtn"] button [data-testid="stIconMaterial"],
[class*="st-key-searchbtn"] button svg { display: none !important; }

/* the collapse arrow: always visible, not only on hover */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarHeader"] button {
    opacity: 1 !important;
    visibility: visible !important;
    display: inline-flex !important;
}
[data-testid="stSidebarCollapseButton"] button {
    background: rgba(128,128,128,0.16) !important;
    border-radius: 8px !important;
}

/* brand: Streamlit reserves a logo slot in the sidebar header itself
   (stLogoSpacer, empty unless st.logo() is used), already on the same flex
   row as the collapse arrow and already vertically centred the same way. A
   negative-margin guess from inside the body content could never really put
   the title on that row; writing into this slot puts it there for real. */
[data-testid="stLogoSpacer"] {
    width: auto !important; height: auto !important;
    display: flex; align-items: center; overflow: visible;
}
[data-testid="stLogoSpacer"]::before {
    content: "Fineprint";
    font-size: 1.7rem; font-weight: 750; letter-spacing: -0.015em;
    padding-left: 0.25rem; white-space: nowrap;
}

/* the chat input, rounder */
div[data-testid="stChatInput"] { border-radius: 16px; }

/* on a fresh chat only, keep the main column tall and drop the example
   questions to its foot so they sit right above the message box. Never do this
   once a conversation is showing: with a long answer, margin-top:auto would
   eat the scroll space and push the question off the top of the view. */
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]:has(.st-key-emptydock) {
    min-height: calc(100vh - 10rem);
}
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]
    > *:has(> .st-key-emptydock) {
    margin-top: auto;
}
/* the "more to try" row after an answer: a little breathing room, normal flow */
.st-key-hintdock { margin-top: 1.4rem; }
.st-key-emptydock { display: flex; flex-direction: column; gap: 0.6rem; }
.st-key-emptydock .fp-hero { margin: 0 0 1.6rem; }
</style>
"""


THINKING_HTML = (
    '<div class="fp-thinking"><span class="d"></span><span class="d"></span>'
    '<span class="d"></span>{label}</div>'
)

# Plain words for how a passage was found, for the answer audit.
RETRIEVER_WORDS = {
    "both": "semantic and keyword search",
    "dense": "semantic search",
    "sparse": "keyword search",
    "none": "neither search (scored after the fact)",
}


@st.cache_resource(show_spinner="Starting Fineprint and loading the local models...")
def load_backend():
    """Open the personal library database and load both models once per process.

    This is deliberately not config.DATABASE_PATH: that one holds the sample
    corpus the evaluation harness is measured against. The app keeps its own
    database so a fresh clone opens on an empty personal workspace instead of
    someone else's 31-document corpus (see config.LIBRARY_DATABASE_PATH).
    """
    # same_thread=False because Streamlit caches this connection across reruns
    # and runs each rerun on a different thread. See rag.db.connect.
    conn = db.connect(config.LIBRARY_DATABASE_PATH, same_thread=False)
    embedder = Embedder()
    # Warm the embedding model now, while the spinner is up, so the first real
    # question does not pay a cold load (which otherwise lands in retrieval_ms).
    try:
        embedder.embed_query("warm up")
    except Exception:  # noqa: BLE001 - a failed warmup must not block the app
        pass
    reranker = None
    if config.RERANK_ENABLED:
        from rag.reranker import Reranker

        reranker = Reranker()
        try:
            reranker.score("warm up", ["warm up"])
        except Exception:  # noqa: BLE001 - a failed warmup must not block the app
            reranker = None
    return conn, Index(conn), embedder, ChatModel(), reranker


def ask_example(question: str) -> None:
    """Callback: queue an example question for the next run."""
    st.session_state.pending = question


_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")


def plain_answer(text: str) -> str:
    """Drop markdown link syntax the model sometimes emits for citations.

    Left in, `[1](1)` renders as a link to a relative path and clicking it
    reloads the app. The citation text is kept, the link is not.
    """
    return _MD_LINK.sub(r"\1", text or "")


# --- conversation store ---------------------------------------------------
# Real chat history: many conversations, each a list of turns, persisted to a
# JSON file next to the database so they survive a restart.


def _load_conversations() -> dict:
    try:
        data = json.loads(CONV_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - a missing or bad file just means no history
        return {}


def _save_conversations() -> None:
    try:
        CONV_PATH.write_text(
            json.dumps(st.session_state.conversations, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    except OSError:
        pass


# Leading question stems, longest first, stripped so the suggested chat name
# reads as a topic ("Federal minimum wage") rather than the raw question.
_QUESTION_STEM = re.compile(
    r"^(?:"
    r"by when can|within how many|how many days do i have to|how many|how much|"
    r"how long|how often|how do i|how does|how|"
    r"what is the|what is|what are the|what are|what|"
    r"when can|when does|when is|when must|when|where|why|who|"
    r"should i|can i|can a|can|do i|does the|does a|is there|are there|under what"
    r")\b[\s,]*",
    re.IGNORECASE,
)
_LEADING_ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)


def _title_from(question: str) -> str:
    topic = " ".join(question.strip().split()).rstrip("?.! ")
    topic = _QUESTION_STEM.sub("", topic)
    topic = _LEADING_ARTICLE.sub("", topic).strip()
    if not topic:
        topic = " ".join(question.strip().split()[:8])
    if len(topic) > 46:
        cut = topic[:46].rsplit(" ", 1)[0]
        topic = (cut or topic[:46]).rstrip() + "…"
    topic = topic[:1].upper() + topic[1:]
    return topic or "New chat"


def _select_conversation(cid: str) -> None:
    """Make cid the active chat, in both session state and the URL.

    The URL is what lets a chat be opened in a new tab: the sidebar links to
    "?chat=<id>", and a fresh tab or a reload reads this back in
    ensure_conversation() to land on the same conversation instead of always
    the most recent one.
    """
    st.session_state.current_id = cid
    st.query_params["chat"] = cid


def _new_conversation() -> None:
    """Start a fresh chat, unless the current one is already empty."""
    convs = st.session_state.conversations
    current = convs.get(st.session_state.get("current_id"))
    if current is not None and not current["turns"]:
        return
    cid = f"c{int(time.time() * 1000)}"
    convs[cid] = {"title": "", "created": time.time(), "turns": []}
    _select_conversation(cid)
    _save_conversations()


def _rename_conversation(cid: str, name: str) -> None:
    convs = st.session_state.conversations
    if cid in convs:
        cleaned = name.strip()
        if cleaned:
            convs[cid]["title"] = cleaned
            _save_conversations()


def _delete_conversation(cid: str) -> None:
    convs = st.session_state.conversations
    convs.pop(cid, None)
    st.session_state.pop("busy", None)
    if st.session_state.get("current_id") == cid:
        if convs:
            _select_conversation(max(convs, key=lambda k: convs[k]["created"]))
        else:
            _new_conversation()
    _save_conversations()


def ensure_conversation() -> None:
    """Pick the active chat: the one in the URL, then the current session's,
    then the most recent, then a brand new one - in that order.

    Checking the URL only when the session has no valid chat of its own
    matters: it is what lets a link opened in a new tab land on that specific
    chat, while an ordinary rerun in an already-open tab (after "New chat" or
    a delete, say) is never overridden by a now-stale "chat" query param.
    """
    st.session_state.setdefault("conversations", _load_conversations())
    convs = st.session_state.conversations
    if st.session_state.get("current_id") not in convs:
        requested = st.query_params.get("chat")
        if requested in convs:
            st.session_state.current_id = requested
        elif convs:
            st.session_state.current_id = max(
                convs, key=lambda k: convs[k]["created"]
            )
        else:
            _new_conversation()
    st.query_params["chat"] = st.session_state.current_id


def current_turns() -> list:
    return st.session_state.conversations[st.session_state.current_id]["turns"]


def _ingest_uploads(conn, index, embedder, files, reranker=None) -> None:
    """Save uploaded documents into the personal library and index them."""
    config.LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for uploaded in files:
        target = config.LIBRARY_DIR / uploaded.name
        target.write_bytes(uploaded.getvalue())
        saved.append(uploaded.name)
    with st.spinner(f"Indexing {len(saved)} file(s)..."):
        report = ingest.ingest_directory(conn, embedder, config.LIBRARY_DIR)
        index.refresh()
        auto_threshold.calibrate_and_store(conn, reranker)
    st.session_state["uploader_gen"] = st.session_state.get("uploader_gen", 0) + 1
    if report.failed:
        names = ", ".join(name for name, _ in report.failed)
        st.warning(f"Could not read: {names}")
    if report.ingested:
        st.toast(f"Added {', '.join(report.ingested)}")
    elif saved:
        st.toast("Already up to date, nothing changed.")
    st.rerun()


def _load_sample_corpus(conn, index, embedder, reranker=None) -> None:
    """Copy the repository's sample corpus into the personal library, on request.

    Reads config.EVAL_CORPUS_DIR (the internship's 31-document evaluation corpus)
    but writes into the app's own database, so it never touches the eval corpus
    or its database, and can be undone with "Clear all documents" below.
    """
    # Embedding 31 documents on the CPU runs for minutes. A bare spinner for
    # that long reads as a hung button, so the progress is shown document by
    # document and the wait is stated up front.
    # Short labels: this also runs from the sidebar, where there is no room.
    bar = st.progress(0.0, text="Indexing locally, a few minutes...")

    def show(done: int, total: int, filename: str) -> None:  # noqa: ARG001
        share = done / total if total else 1.0
        label = f"{done} / {total} documents"
        bar.progress(share, text=label)

    report = ingest.ingest_directory(
        conn, embedder, config.EVAL_CORPUS_DIR, on_progress=show
    )
    bar.progress(1.0, text="Calibrating the threshold...")
    index.refresh()
    auto_threshold.calibrate_and_store(conn, reranker)
    bar.empty()
    st.toast(f"Sample corpus loaded: {report.summary()}")
    st.rerun()


def _remove_document(conn, index, filename: str, reranker=None) -> None:
    """Delete one document from the personal library.

    The file on disk is removed along with the database row. Leaving it
    behind meant the next upload silently re-ingested it: ingest_directory
    walks the whole library folder, and a file with no matching row just
    looks like a new document to it.
    """
    if db.delete_document(conn, filename):
        db.rebuild_fts(conn)
        conn.commit()
        index.refresh()
        auto_threshold.calibrate_and_store(conn, reranker)
        (config.LIBRARY_DIR / filename).unlink(missing_ok=True)
    st.toast(f"Removed {filename}")
    st.rerun()


def _clear_library(conn, index) -> None:
    """Delete every document from the personal library, files included.

    Same reasoning as _remove_document: a file left on disk after its row is
    gone is indistinguishable from a new upload, and the next ingest silently
    re-embeds it.
    """
    removed = db.delete_all_documents(conn)
    db.rebuild_fts(conn)
    conn.commit()
    index.refresh()
    db.delete_meta(conn, db.META_RERANK_THRESHOLD)
    for path in loaders.discover_documents(config.LIBRARY_DIR):
        path.unlink(missing_ok=True)
    st.session_state.pop("confirm_clear", None)
    st.toast(f"Removed {removed} document(s). Your library is empty again.")
    st.rerun()


def record_turn(turn: dict) -> None:
    conv = st.session_state.conversations[st.session_state.current_id]
    conv["turns"].append(turn)
    if not conv["title"]:
        conv["title"] = _title_from(turn["question"])
    _save_conversations()


# --- sidebar ----------------------------------------------------------------

# The three sidebar panels (Documents, How Fineprint works, Advanced) behave
# as an accordion: opening one closes the others, so the sidebar never grows
# to show two long panels at once.
_ACCORDION_KEYS = ("exp_documents", "exp_how_it_works", "exp_advanced")


def _accordion_close_others(opened_key: str) -> None:
    """on_change callback: when one panel opens, collapse the rest.

    Runs after Streamlit has already written the new (just-toggled) value
    into st.session_state[opened_key], so a True here means the user just
    expanded this panel.
    """
    if st.session_state.get(opened_key):
        for key in _ACCORDION_KEYS:
            if key != opened_key:
                st.session_state[key] = False


def render_sidebar(
    conn, index, embedder, reranker=None
) -> tuple[Mode, float, float]:
    """A real chatbot sidebar: new chat, search, the list of conversations, and
    settings tucked in above the account chip at the bottom. Adding documents
    happens at the chat box itself; this only lists and removes them."""
    threshold = config.RELEVANCE_THRESHOLD
    # Auto-calibrated per library (see rag/threshold.py), recomputed after every
    # upload, removal or corpus load; config's number is only the fallback
    # before anything has been calibrated yet (an empty library, or one where
    # calibration could not find a signal).
    stored_rerank_threshold = db.get_meta(conn, db.META_RERANK_THRESHOLD)
    auto_calibrated = stored_rerank_threshold is not None
    rerank_threshold = (
        float(stored_rerank_threshold)
        if auto_calibrated
        else config.RERANK_RELEVANCE_THRESHOLD
    )
    mode = Mode(config.RETRIEVAL_MODE)
    convs = st.session_state.conversations

    with st.sidebar:
        # The title itself is drawn in CSS, into the header's own logo slot
        # (see [data-testid="stLogoSpacer"]::before in STYLE) - not written
        # here, so nothing below has to guess where the header row is.

        # search: a magnifier toggle that reveals an inline box. Inline (not a
        # popover) so it keeps focus and its text across the reruns that typing
        # triggers, which is what made the popover version feel broken.
        if st.button("Search chats", key="searchbtn", use_container_width=True):
            st.session_state.search_open = not st.session_state.get(
                "search_open", False
            )
        query = ""
        if st.session_state.get("search_open"):
            query = st.text_input(
                "search", key="convsearch", placeholder="Search chats",
                label_visibility="collapsed",
            ).strip()

        if st.button("New chat", key="newchat", use_container_width=True):
            _new_conversation()
            st.session_state.pop("busy", None)
            st.rerun()

        current_id = st.session_state.current_id

        # the list of past chats (any with messages). The current one is
        # highlighted; a blank new chat only appears once it has a turn.
        ordered = sorted(
            (kv for kv in convs.items() if kv[1]["turns"]),
            key=lambda kv: kv[1]["created"], reverse=True,
        )
        if ordered:
            st.markdown(
                '<div class="fp-label">Chats</div>', unsafe_allow_html=True
            )
        shown = 0
        with st.container(key="convlist"):
            for cid, conv in ordered:
                title = conv["title"] or "Untitled chat"
                if query and query.lower() not in title.lower():
                    continue
                shown += 1
                is_current = cid == current_id
                with st.container(key=f"convrow_{cid}"):
                    pick, menu = st.columns([0.83, 0.17], gap="small")
                    # A real link, not a button: right-click gives the
                    # browser's own "open in new tab", which no st.button
                    # can offer since it has no URL to open. target="_self"
                    # is needed too - without it a left click opened a new
                    # tab as well, which defeats instant chat switching.
                    pick.markdown(
                        f'<a class="fp-convlink{" active" if is_current else ""}" '
                        f'href="?chat={cid}" target="_self">{html.escape(title)}</a>',
                        unsafe_allow_html=True,
                    )
                    with menu.popover("⋮", key=f"convmenu_{cid}",
                                      use_container_width=True):
                        new_name = st.text_input(
                            "Rename chat", value=conv["title"],
                            key=f"rename_{cid}", placeholder="Chat name",
                            label_visibility="collapsed",
                        )
                        act_rename, act_delete = st.columns(2)
                        if act_rename.button(
                            "Rename", key=f"dorename_{cid}", use_container_width=True
                        ):
                            _rename_conversation(cid, new_name)
                            st.rerun()
                        if act_delete.button(
                            "Delete", key=f"dodelete_{cid}", use_container_width=True
                        ):
                            _delete_conversation(cid)
                            st.rerun()
            if shown == 0:
                st.caption("No chats match." if query else "No saved chats yet.")

        # settings sit just above the account chip, pushed down. Adding
        # documents happens at the chat box's own attach button, not here;
        # this is only for seeing what is loaded and removing it.
        with st.container(key="sidebarfoot"):
            doc_count = db.count_documents(conn)
            with st.expander(
                f"{doc_count} document(s)" if doc_count else "Documents",
                key="exp_documents", on_change=_accordion_close_others,
                args=("exp_documents",),
            ):
                render_document_panel(conn, index, embedder, reranker)

            with st.expander(
                "How Fineprint works",
                key="exp_how_it_works", on_change=_accordion_close_others,
                args=("exp_how_it_works",),
            ):
                st.caption(
                    "It answers questions about the loaded documents, quoting "
                    "the clause the answer comes from. If nothing in the "
                    "documents answers the question, or the question asks for "
                    "advice, it says so instead of guessing. Everything runs on "
                    "this machine, so an answer usually takes 30 to 60 seconds."
                )

            with st.expander(
                "Advanced",
                key="exp_advanced", on_change=_accordion_close_others,
                args=("exp_advanced",),
            ):
                mode_values = [item.value for item in Mode]
                mode = Mode(
                    st.selectbox(
                        "Retrieval strategy",
                        options=mode_values,
                        index=mode_values.index(config.RETRIEVAL_MODE),
                        help="Dense scored the highest passage-level retrieval "
                        "on this corpus; see docs/adr/0002.",
                    )
                )
                if reranker is None:
                    threshold = st.slider(
                        "Relevance threshold",
                        min_value=0.0, max_value=1.0,
                        value=float(config.RELEVANCE_THRESHOLD), step=0.01,
                        help="Below this dense similarity the assistant refuses.",
                    )
                else:
                    st.caption(
                        (
                            "Relevance gate: auto-calibrated to your library at "
                            if auto_calibrated
                            else "Relevance gate: no calibration yet, using the "
                            "project default of "
                        )
                        + f"**{rerank_threshold:.1f}**."
                    )
                    rerank_threshold = st.slider(
                        "Relevance gate override (cross-encoder score)",
                        min_value=-12.0, max_value=12.0,
                        value=float(rerank_threshold), step=0.1,
                        help="Below this cross-encoder score the assistant "
                        "refuses. Recalibrated automatically after every "
                        "upload, removal or corpus load (rag/threshold.py), "
                        "so this rarely needs a manual touch - lower it "
                        "yourself only if an answer you can see sitting right "
                        "there in the sources still gets refused.",
                    )
                st.caption(
                    f"Models: `{config.CHAT_MODEL}`, `{config.EMBEDDING_MODEL}` "
                    "via Microsoft Foundry Local. Add documents with the "
                    "attach icon in the chat box; that is the only supported "
                    "way in, since the CLI's `rag.ingest` always targets the "
                    "sample corpus database, not this app's library."
                )
                if st.button("Refresh index", use_container_width=True):
                    index.refresh()
                    st.rerun()

            st.markdown(
                '<div class="fp-user"><span class="fp-user-dot">U</span>'
                "<span><b>User</b><br><span class='sub'>Local session</span>"
                "</span></div>",
                unsafe_allow_html=True,
            )

    return mode, threshold, rerank_threshold


# --- answer rendering -----------------------------------------------------


def _answer_to_turn(question: str, answer: pipeline.Answer, audit) -> dict:
    """A plain, JSON-safe record of one turn, for rendering and for the store."""
    return {
        "question": question,
        "text": answer.text,
        "timestamp": time.time(),
        "outcome": answer.outcome.value,
        "answered": answer.answered,
        "mode": answer.mode.value,
        "threshold": answer.threshold,
        "rerank_threshold": answer.rerank_threshold,
        "retrieval_ms": answer.retrieval_ms,
        "generation_ms": answer.generation_ms,
        "generation_retried": answer.generation_retried,
        "best_dense_score": answer.best_dense_score,
        "best_rerank_score": answer.best_rerank_score,
        "sources": [
            {
                "citation": s.chunk.citation(),
                "content": s.chunk.content,
                "dense_score": s.dense_score,
                "sparse_rank": s.sparse_rank,
                "fused_score": s.fused_score,
            }
            for s in answer.sources
        ],
        "supported_fraction": audit.supported_fraction if audit else None,
        "audit": (
            [
                {
                    "sentence": t.sentence,
                    "citation": t.source_citation,
                    "similarity": t.similarity,
                    "retriever": t.retriever,
                    "supported": t.supported,
                }
                for t in audit.traces
            ]
            if audit
            else None
        ),
    }


def outcome_note(turn: dict) -> None:
    """Explain why a question was refused, when it was."""
    outcome = turn["outcome"]
    if outcome == "not_in_corpus":
        if turn.get("best_rerank_score") is not None:
            gate = turn.get("rerank_threshold", config.RERANK_RELEVANCE_THRESHOLD)
            detail = (
                f"best passage scored {turn['best_rerank_score']:.1f}, needs "
                f"{gate:.1f}"
            )
        else:
            detail = (
                f"best match {turn['best_dense_score']:.2f}, needs "
                f"{turn['threshold']:.2f}"
            )
        st.info(
            "Not in your documents. Nothing was close enough to answer from "
            f"({detail}). The model was not called."
        )
    elif outcome == "advice_refused":
        st.warning(
            "This asks for advice. Fineprint reports what a document says, not "
            "what you should do."
        )
    elif outcome == "generation_failed":
        st.error("The local model returned an empty answer. Ask again.")


def _source_kind(filename: str) -> str:
    """A short file-type tag for a citation chip: PDF, DOCX, TXT, MD, ..."""
    suffix = Path(filename).suffix.lstrip(".").upper()
    return suffix or "DOC"


def _short_source_name(citation: str, limit: int = 20) -> str:
    """A citation's filename, without its heading, truncated for a chip."""
    filename = citation.split(",", 1)[0].strip()
    stem = Path(filename).stem
    return stem if len(stem) <= limit else stem[: limit - 1].rstrip() + "…"


def render_sources(turn: dict, turn_index: int) -> None:
    """One quiet "Sources" trigger under the answer, not a row of chips: three
    of those never sat evenly, and a document icon plus a dropdown reads as
    one control rather than three competing for a place to line up. Each
    source inside opens and closes on its own, independent of the others.
    The "how this answer was built" toggle sits beside it in the same row,
    filling what Sources leaves of the answer's own text width, rather than
    stacking as a second, separately-sized block underneath.
    """
    sources = turn["sources"]
    if not sources:
        return
    # One popover, not two side by side: Streamlit gives each st.popover its
    # own open/closed state with no way for one to force the other shut, so
    # two independent triggers could both end up open and overlap. Nesting
    # "how this answer was built" inside the Sources panel instead of beside
    # it removes the second floating panel entirely - nothing left to clash.
    with st.container(key=f"srcrow_{turn_index}"):
        with st.popover(f"Sources ({len(sources)})", key=f"srcpop_{turn_index}"):
            for position, src in enumerate(sources, start=1):
                label = (
                    f"{position}. {_source_kind(src['citation'])} · "
                    f"{_short_source_name(src['citation'], limit=34)}"
                )
                with st.expander(label, expanded=False):
                    st.caption(src["citation"])
                    st.write(src["content"])
                    keyword = (
                        f"keyword rank {src['sparse_rank']}"
                        if src["sparse_rank"]
                        else "not a keyword match"
                    )
                    st.markdown(
                        f'<div class="fp-srcmeta">semantic match '
                        f"{src['dense_score']:.2f} &nbsp;&middot;&nbsp; {keyword} "
                        f"&nbsp;&middot;&nbsp; fused score "
                        f"{src['fused_score']:.3f}</div>",
                        unsafe_allow_html=True,
                    )

            traces = turn.get("audit")
            if traces:
                st.divider()
                with st.expander("How this answer was built"):
                    st.caption(
                        f"{turn['supported_fraction']:.0%} of sentences trace "
                        "to a source. Each sentence of the answer is compared "
                        "back to the sources above: a strong match closely "
                        "restates one of them, a weak match goes beyond what "
                        "was retrieved."
                    )
                    for index, trace in enumerate(traces, start=1):
                        strength = "Strong match" if trace["supported"] else "Weak match"
                        finder = RETRIEVER_WORDS.get(
                            trace["retriever"], trace["retriever"]
                        )
                        st.markdown(f"**{index}.** {trace['sentence']}")
                        st.caption(
                            f"{strength} ({trace['similarity']:.2f}) to "
                            f"*{trace['citation']}*, found by {finder}."
                        )


def _humanise_ms(value: float) -> str:
    """A duration that reads well: milliseconds under a second, seconds above."""
    return f"{value:.0f} ms" if value < 1000 else f"{value / 1000:.1f} s"


def render_metrics(turn: dict) -> None:
    """A short, readable line of timings and the models used."""
    if turn["generation_ms"]:
        line = (
            f"Retrieved passages in <b>{_humanise_ms(turn['retrieval_ms'])}</b>, "
            f"wrote the answer in <b>{_humanise_ms(turn['generation_ms'])}</b>. "
            f"{turn['mode']} retrieval, {config.CHAT_MODEL} on CPU"
            + (", retried once" if turn["generation_retried"] else "")
            + "."
        )
    else:
        line = (
            f"Decided in <b>{_humanise_ms(turn['retrieval_ms'])}</b> without "
            "calling the model."
        )
    st.markdown(f'<div class="fp-metrics">{line}</div>', unsafe_allow_html=True)


def _relative_time(timestamp: float | None) -> str:
    """A short "how long ago" label, falling back to a date past a week."""
    if not timestamp:
        return ""
    delta = time.time() - timestamp
    if delta < 60:
        return "just now"
    minutes = int(delta // 60)
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = int(minutes // 60)
    if hours < 24:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    days = int(hours // 24)
    if days < 7:
        return f"{days} day{'s' if days != 1 else ''} ago"
    # %-d (no leading zero) is POSIX-only; %d works on Windows too, so the
    # leading zero on early-month days is stripped by hand instead.
    stamp = datetime.fromtimestamp(timestamp).strftime("%b %d")
    return stamp.replace(" 0", " ")


# Colours are hardcoded rather than read from the CSS variables above: this
# renders inside components.html's own iframe, a separate document that
# cannot see the parent page's stylesheet or its custom properties.
_FOOTER_HTML = """
<div style="display:flex;align-items:center;gap:8px;height:22px;
    font-family:'Source Sans Pro',system-ui,-apple-system,sans-serif;">
  <button id="fpcopy-{key}" title="Copy answer" aria-label="Copy answer"
      style="display:inline-flex;align-items:center;justify-content:center;
      background:none;cursor:pointer;border:1px solid rgba(255,255,255,0.16);
      border-radius:7px;width:24px;height:24px;padding:0;color:#9a9ea6;">
    <svg id="fpcopy-icon-{key}" width="12" height="12" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2"></rect>
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
    </svg>
  </button>
  <span style="color:#72767d;font-size:12px;">{when}</span>
</div>
<script>
(function() {{
  var btn = document.getElementById("fpcopy-{key}");
  var icon = document.getElementById("fpcopy-icon-{key}");
  var checkmark = '<polyline points="20 6 9 17 4 12"></polyline>';
  var clipboard = '<rect x="9" y="9" width="13" height="13" rx="2"></rect>' +
      '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>';
  function flash() {{
    icon.innerHTML = checkmark;
    btn.title = "Copied";
    setTimeout(function() {{
      icon.innerHTML = clipboard;
      btn.title = "Copy answer";
    }}, 1500);
  }}
  function legacyCopy(value) {{
    // Falls back to this when the Clipboard API is unavailable or the
    // embedding page's permissions policy denies it (both seen in the wild
    // for a components.html iframe) - execCommand needs no such grant.
    var area = document.createElement("textarea");
    area.value = value;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.focus();
    area.select();
    try {{ document.execCommand("copy"); }} catch (err) {{ /* no-op */ }}
    document.body.removeChild(area);
  }}
  btn.addEventListener("click", function() {{
    var value = {payload};
    if (navigator.clipboard && navigator.clipboard.writeText) {{
      navigator.clipboard.writeText(value).then(flash, function() {{
        legacyCopy(value);
        flash();
      }});
    }} else {{
      legacyCopy(value);
      flash();
    }}
  }});
}})();
</script>
"""


def render_footer(turn: dict, turn_index: int) -> None:
    """Copy-to-clipboard and a relative timestamp, under every answer.

    A real clipboard write needs a script that actually executes, which
    st.markdown's sanitized HTML will not run; components.html renders in its
    own iframe where scripts do run, at the cost of hardcoding colours since
    that iframe cannot see this page's CSS.

    A turn answered before this feature existed has no timestamp of its own.
    Rather than leave the time blank or invent one, it falls back to the
    conversation's own "created" timestamp - real (it is when that chat
    actually started), just less precise than a per-turn one for a chat with
    several turns in it.
    """
    timestamp = turn.get("timestamp")
    if not timestamp:
        conv = st.session_state.conversations.get(st.session_state.current_id)
        timestamp = conv.get("created") if conv else None
    html_code = _FOOTER_HTML.format(
        key=turn_index,
        when=html.escape(_relative_time(timestamp)),
        payload=json.dumps(turn["text"]),
    )
    components.html(html_code, height=26)


def render_document_panel(conn, index, embedder, reranker=None) -> None:
    """Sidebar document manager: the list of what is loaded, plus removing it.

    Adding documents happens at the chat box itself (its native attach
    button), not here; this is for seeing what is loaded and taking it back
    out, which a plain attach icon has no room for.
    """
    with st.container(key="docpanel"):
        documents = db.list_documents(conn)
        if documents:
            st.caption(f"{len(documents)} document(s) in your library")
            for entry in documents:
                row_text, row_menu = st.columns([0.86, 0.14], gap="small")
                row_text.markdown(
                    f"**{entry['title']}**  \n"
                    f"<span class='fp-doc-sub'>`{entry['filename']}` · "
                    f"{entry['passages']} passages</span>",
                    unsafe_allow_html=True,
                )
                with row_menu.popover(
                    "⋮", key=f"docmenu_{entry['filename']}", use_container_width=True
                ):
                    if st.button(
                        "Delete", key=f"dodeletedoc_{entry['filename']}",
                        use_container_width=True,
                    ):
                        _remove_document(conn, index, entry["filename"], reranker)

        st.divider()
        if st.button(
            "Load sample corpus", key="loadsample", use_container_width=True,
            help="Adds the 31 public-domain regulations the internship "
            "evaluation is measured against, without touching your own "
            "documents.",
        ):
            _load_sample_corpus(conn, index, embedder, reranker)

        if documents:
            if st.session_state.get("confirm_clear"):
                st.warning(
                    "Remove all documents from your library? This cannot be "
                    "undone."
                )
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button(
                    "Yes, clear", key="doclear", use_container_width=True
                ):
                    _clear_library(conn, index)
                if cancel_col.button(
                    "Cancel", key="cancelclear", use_container_width=True
                ):
                    st.session_state.pop("confirm_clear", None)
                    st.rerun()
            elif st.button(
                "Clear all documents", key="askclear", use_container_width=True
            ):
                st.session_state["confirm_clear"] = True
                st.rerun()


def render_turn(turn: dict, turn_index: int) -> None:
    """Render one finished question and answer."""
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        if turn["answered"]:
            st.markdown('<div class="fp-eyebrow">Answer</div>', unsafe_allow_html=True)
        st.write(turn["text"])
        outcome_note(turn)
        render_sources(turn, turn_index)
        render_metrics(turn)
        render_footer(turn, turn_index)


def answer_question(
    conn, index, embedder, chat_model, question, mode, threshold, reranker=None,
    rerank_threshold: float = config.RERANK_RELEVANCE_THRESHOLD,
):
    """Run one question, stream it into a chat bubble, and store it."""
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.markdown(
            THINKING_HTML.format(label="Searching your documents"),
            unsafe_allow_html=True,
        )
        answer, stream = pipeline.ask_streaming(
            conn, index, embedder, chat_model, question,
            mode=mode, threshold=threshold, reranker=reranker,
            rerank_threshold=rerank_threshold,
        )
        if answer.answered:
            thinking.markdown(
                THINKING_HTML.format(label="Writing the answer locally"),
                unsafe_allow_html=True,
            )
            text = st.write_stream(stream)
            thinking.empty()
        else:
            thinking.empty()
            text = "".join(stream)
            st.write(text)
        answer.text = plain_answer(text or answer.text)

        audit = None
        if answer.answered and answer.sources:
            audit = audit_answer(answer.text, answer.sources, embedder)

        turn = _answer_to_turn(question, answer, audit)
        outcome_note(turn)
        turn_index = len(current_turns())
        render_sources(turn, turn_index)
        render_metrics(turn)
        render_footer(turn, turn_index)

    pipeline.log_answer(answer, config.QUERY_LOG_PATH)
    record_turn(turn)


# --- page ---------------------------------------------------------------------


def main() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)
    ensure_conversation()

    conn, index, embedder, chat_model, reranker = load_backend()
    chunk_count = db.count_chunks(conn)
    mode, threshold, rerank_threshold = render_sidebar(conn, index, embedder, reranker)

    turns = current_turns()
    for turn_index, turn in enumerate(turns):
        render_turn(turn, turn_index)

    # A question in flight: show it working, keep the input visibly disabled.
    in_flight = st.session_state.get("busy")
    if in_flight:
        st.chat_input(
            "Fineprint is working. The model runs on CPU, so a hard question "
            "can take a few minutes...",
            disabled=True,
        )
        answer_question(
            conn, index, embedder, chat_model, in_flight, mode, threshold, reranker,
            rerank_threshold=rerank_threshold,
        )
        st.session_state.busy = None
        st.rerun()
        return

    if chunk_count == 0:
        # An empty library is the first thing a new user sees, so it gets the
        # same welcome screen as an empty chat rather than a lone banner. The
        # sample corpus is not offered here: this app is for your own
        # documents, and the corpus is an evaluation fixture that lives behind
        # Documents in the sidebar for anyone who wants to try it quickly.
        with st.container(key="emptydock"):
            st.markdown(
                '<div class="fp-hero"><b>Add a document to begin</b>'
                "<span>Click the <b>+</b> in the message box below to attach "
                "PDF, DOCX, TXT or MD files, or drop them straight onto it. "
                "They are indexed on this machine and nothing leaves it."
                "</span></div>",
                unsafe_allow_html=True,
            )

    else:
        show_examples = _library_is_sample_corpus_only(conn)
        if not turns:
            # A fresh chat: a modest greeting (the app name lives in the
            # sidebar) and, only over the sample corpus, the example
            # questions written for it, dropped to sit above the input.
            with st.container(key="emptydock"):
                st.markdown(
                    '<div class="fp-hero"><b>Ask about the documents</b>'
                    "<span>Every answer is quoted from a clause. If the "
                    "documents do not cover it, Fineprint says so instead of "
                    "guessing.</span></div>",
                    unsafe_allow_html=True,
                )
                if show_examples:
                    st.markdown(
                        '<div class="fp-label">Try one</div>',
                        unsafe_allow_html=True,
                    )
                    for start in range(0, len(EXAMPLE_QUESTIONS), 2):
                        columns = st.columns(2, gap="medium")
                        for offset, column in enumerate(columns):
                            example = EXAMPLE_QUESTIONS[start + offset]
                            column.button(
                                example,
                                key=f"sugg_{start + offset}",
                                on_click=ask_example,
                                args=(example,),
                                use_container_width=True,
                            )
        elif show_examples:
            # After the first answer: a faded row of examples, dropped to
            # just above the message box. Only for the sample corpus, same
            # reasoning as above.
            with st.container(key="hintdock"):
                st.markdown(
                    '<div class="fp-label" style="margin-bottom:0.3rem">'
                    "More to try</div>",
                    unsafe_allow_html=True,
                )
                columns = st.columns(3)
                for offset, column in enumerate(columns):
                    example = EXAMPLE_QUESTIONS[offset]
                    column.button(
                        example,
                        key=f"hint_{offset}",
                        on_click=ask_example,
                        args=(example,),
                        use_container_width=True,
                    )

    # The attach icon lives inside the input itself (left side, native to
    # st.chat_input): click it for a file browser or drop files straight onto
    # it. A submission carries text, files, or both.
    file_types = [suffix.lstrip(".") for suffix in config.SUPPORTED_SUFFIXES]
    submission = st.chat_input(
        "Ask about your documents" if chunk_count else "Add documents to get started",
        accept_file="multiple",
        file_type=file_types,
    )

    question = None
    if submission:
        if isinstance(submission, str):
            files, text = [], submission.strip()
        else:
            files = list(getattr(submission, "files", None) or [])
            text = (getattr(submission, "text", None) or "").strip()
        if files:
            # Show the message as sent before indexing starts, not after: the
            # spinner inside _ingest_uploads can run for a while (a big file,
            # or several), and jumping straight to it with no bubble first
            # looked like the attachment had gone nowhere.
            with st.chat_message("user"):
                st.write(text or "(no question, just adding documents)")
                st.caption(
                    "Attaching: " + ", ".join(uploaded.name for uploaded in files)
                )
            if text:
                # Ingesting reruns the page immediately; stash the question so
                # it is asked on the run right after the upload finishes.
                st.session_state["pending"] = text
            _ingest_uploads(conn, index, embedder, files, reranker)
        else:
            question = text or None

    question = question or st.session_state.pop("pending", None)
    if question:
        st.session_state.busy = question
        st.rerun()


if __name__ == "__main__":
    main()
