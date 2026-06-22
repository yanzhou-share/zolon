(function() {
  const API_URL = (window.AI_CHAT_API_URL || 'http://localhost:8000').replace(/\/+$/, '');
  const API_KEY = window.AI_CHAT_API_KEY || '';
  const ROOT_ID = 'ai-chat-widget';
  const STORAGE_KEY = 'ai_chat_session';
  const HISTORY_KEY = 'ai_chat_history';

  const css = `
#${ROOT_ID} * { margin: 0; padding: 0; box-sizing: border-box; }
#${ROOT_ID} { position: fixed; bottom: 24px; right: 24px; z-index: 999999; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
#${ROOT_ID} .acw-fab { width: 56px; height: 56px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none; cursor: pointer; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 24px rgba(102,126,234,0.45); transition: transform 0.3s, box-shadow 0.3s; }
#${ROOT_ID} .acw-fab:hover { transform: scale(1.08); box-shadow: 0 6px 32px rgba(102,126,234,0.6); }
#${ROOT_ID} .acw-fab svg { width: 28px; height: 28px; fill: #fff; }
#${ROOT_ID} .acw-fab .acw-badge { position: absolute; top: -2px; right: -2px; width: 16px; height: 16px; border-radius: 50%; background: #ef4444; border: 2px solid #fff; display: none; }
#${ROOT_ID} .acw-panel { position: absolute; bottom: 70px; right: 0; width: 380px; height: 520px; background: #fff; border-radius: 16px; box-shadow: 0 8px 48px rgba(0,0,0,0.15); display: flex; flex-direction: column; overflow: hidden; opacity: 0; transform: translateY(12px) scale(0.95); pointer-events: none; transition: opacity 0.25s ease, transform 0.25s ease; }
#${ROOT_ID} .acw-panel.acw-open { opacity: 1; transform: translateY(0) scale(1); pointer-events: auto; }
#${ROOT_ID} .acw-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #fff; padding: 16px 18px; display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
#${ROOT_ID} .acw-header-avatar { width: 38px; height: 38px; border-radius: 50%; background: rgba(255,255,255,0.2); display: flex; align-items: center; justify-content: center; font-size: 20px; }
#${ROOT_ID} .acw-header-info { flex: 1; }
#${ROOT_ID} .acw-header-title { font-size: 15px; font-weight: 600; }
#${ROOT_ID} .acw-header-status { font-size: 12px; opacity: 0.85; display: flex; align-items: center; gap: 4px; }
#${ROOT_ID} .acw-header-status::before { content: ''; width: 7px; height: 7px; border-radius: 50%; background: #4ade80; display: inline-block; }
#${ROOT_ID} .acw-close { background: none; border: none; color: #fff; cursor: pointer; padding: 4px; font-size: 20px; line-height: 1; opacity: 0.8; transition: opacity 0.2s; }
#${ROOT_ID} .acw-close:hover { opacity: 1; }
#${ROOT_ID} .acw-messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; background: #f8f9fb; }
#${ROOT_ID} .acw-messages::-webkit-scrollbar { width: 4px; }
#${ROOT_ID} .acw-messages::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 2px; }
#${ROOT_ID} .acw-msg { max-width: 85%; display: flex; flex-direction: column; gap: 4px; }
#${ROOT_ID} .acw-msg-user { align-self: flex-end; }
#${ROOT_ID} .acw-msg-bot { align-self: flex-start; }
#${ROOT_ID} .acw-msg-bubble { padding: 10px 14px; border-radius: 14px; font-size: 14px; line-height: 1.6; word-break: break-word; white-space: pre-wrap; }
#${ROOT_ID} .acw-msg-user .acw-msg-bubble { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #fff; border-bottom-right-radius: 4px; }
#${ROOT_ID} .acw-msg-bot .acw-msg-bubble { background: #fff; color: #1f2937; border-bottom-left-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
#${ROOT_ID} .acw-msg-meta { font-size: 11px; color: #9ca3af; padding: 0 4px; }
#${ROOT_ID} .acw-msg-bot .acw-msg-meta { padding-left: 4px; }
#${ROOT_ID} .acw-msg-user .acw-msg-meta { text-align: right; padding-right: 4px; }
#${ROOT_ID} .acw-typing { display: flex; gap: 4px; padding: 12px 16px; align-self: flex-start; }
#${ROOT_ID} .acw-typing span { width: 7px; height: 7px; border-radius: 50%; background: #9ca3af; animation: acw-bounce 1.4s infinite ease-in-out; }
#${ROOT_ID} .acw-typing span:nth-child(2) { animation-delay: 0.16s; }
#${ROOT_ID} .acw-typing span:nth-child(3) { animation-delay: 0.32s; }
@keyframes acw-bounce { 0%,80%,100% { transform: scale(0.6); opacity: 0.4; } 40% { transform: scale(1); opacity: 1; } }
#${ROOT_ID} .acw-handover-notice { background: #fef3c7; color: #92400e; padding: 8px 14px; border-radius: 8px; font-size: 12px; text-align: center; margin: 4px 0; }
#${ROOT_ID} .acw-input-area { padding: 12px 16px; border-top: 1px solid #e5e7eb; display: flex; gap: 8px; align-items: flex-end; background: #fff; flex-shrink: 0; }
#${ROOT_ID} .acw-input { flex: 1; border: 1px solid #e5e7eb; border-radius: 12px; padding: 10px 14px; font-size: 14px; line-height: 1.5; resize: none; outline: none; font-family: inherit; max-height: 80px; min-height: 40px; transition: border-color 0.2s; }
#${ROOT_ID} .acw-input:focus { border-color: #667eea; }
#${ROOT_ID} .acw-send { width: 40px; height: 40px; border-radius: 50%; border: none; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); cursor: pointer; display: flex; align-items: center; justify-content: center; flex-shrink: 0; transition: opacity 0.2s, transform 0.2s; }
#${ROOT_ID} .acw-send:hover { transform: scale(1.05); }
#${ROOT_ID} .acw-send:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
#${ROOT_ID} .acw-send svg { width: 18px; height: 18px; fill: #fff; }
#${ROOT_ID} .acw-welcome { text-align: center; padding: 32px 20px; color: #6b7280; }
#${ROOT_ID} .acw-welcome-icon { font-size: 40px; margin-bottom: 12px; }
#${ROOT_ID} .acw-welcome h3 { font-size: 16px; color: #374151; margin-bottom: 6px; }
#${ROOT_ID} .acw-welcome p { font-size: 13px; line-height: 1.6; }
@media (max-width: 480px) {
  #${ROOT_ID} .acw-panel { position: fixed; bottom: 0; right: 0; left: 0; top: 0; width: 100%; height: 100%; border-radius: 0; }
  #${ROOT_ID} { bottom: 16px; right: 16px; }
}`;

  function injectStyles() {
    const style = document.createElement('style');
    style.textContent = css;
    document.head.appendChild(style);
  }

  function createRoot() {
    if (!document.getElementById(ROOT_ID)) {
      const root = document.createElement('div');
      root.id = ROOT_ID;
      document.body.appendChild(root);
    }
  }

  let sessionId = localStorage.getItem(STORAGE_KEY);
  if (!sessionId) {
    sessionId = 'w-' + crypto.randomUUID();
    localStorage.setItem(STORAGE_KEY, sessionId);
  }

  let chatHistory = [];
  try {
    const saved = localStorage.getItem(HISTORY_KEY + '_' + sessionId);
    if (saved) chatHistory = JSON.parse(saved);
  } catch(e) {}

  let isOpen = false;
  let isStreaming = false;
  let panelEl, messagesEl, inputEl, sendBtn, fabEl, badgeEl;
  let streamAbort = null;

  function saveHistory() {
    try { localStorage.setItem(HISTORY_KEY + '_' + sessionId, JSON.stringify(chatHistory)); } catch(e) {}
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function formatTime() {
    return new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
  }

  function renderMessages() {
    const welcomeHtml = chatHistory.length === 0 ? '<div class="acw-welcome"><div class="acw-welcome-icon">&#x1F4AC;</div><h3>&#x667A;&#x80FD;&#x5BA2;&#x670D;</h3><p>&#x6B22;&#x8FCE;&#x54A8;&#x8BE2;&#xFF0C;&#x6211;&#x5C06;&#x4E3A;&#x60A8;&#x63D0;&#x4F9B;&#x4E13;&#x4E1A;&#x7684;&#x670D;&#x52A1;&#x3002;</p></div>' : '';

    const msgsHtml = chatHistory.map(msg => {
      if (msg.role === 'user') {
        return '<div class="acw-msg acw-msg-user"><div class="acw-msg-bubble">' + escapeHtml(msg.content) + '</div><div class="acw-msg-meta">' + (msg.time || '') + '</div></div>';
      } else {
        const handoverNotice = msg.status === 'human' ? '<div class="acw-handover-notice">&#x26A0;&#xFE0F; &#x5C06;&#x4E3A;&#x60A8;&#x8F6C;&#x63A5;&#x4EBA;&#x5DE5;&#x5BA2;&#x670D;</div>' : '';
        return '<div class="acw-msg acw-msg-bot">' + handoverNotice + '<div class="acw-msg-bubble">' + escapeHtml(msg.content) + '</div><div class="acw-msg-meta">' + (msg.time || '') + '</div></div>';
      }
    }).join('');

    messagesEl.innerHTML = welcomeHtml + msgsHtml;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function addTypingIndicator() {
    const el = document.createElement('div');
    el.className = 'acw-typing';
    el.id = 'acw-typing';
    el.innerHTML = '<span></span><span></span><span></span>';
    messagesEl.appendChild(el);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function removeTypingIndicator() {
    const el = document.getElementById('acw-typing');
    if (el) el.remove();
  }

  function updateBadge(show) {
    if (badgeEl) badgeEl.style.display = show ? 'block' : 'none';
  }

  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || isStreaming) return;

    isStreaming = true;
    sendBtn.disabled = true;
    inputEl.value = '';
    inputEl.style.height = 'auto';

    chatHistory.push({ role: 'user', content: text, time: formatTime() });
    saveHistory();
    renderMessages();

    addTypingIndicator();

    const historyForApi = chatHistory.slice(0, -1).map(m => ({
      role: m.role === 'bot' ? 'assistant' : m.role,
      content: m.content
    }));

    let fullReply = '';
    let meta = {};
    let botMsgIndex = -1;

    streamAbort = new AbortController();

    try {
      const headers = { 'Content-Type': 'application/json' };
      if (API_KEY) headers['X-API-Key'] = API_KEY;
      const response = await fetch(API_URL + '/chat/stream', {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({
          session_id: sessionId,
          message: text,
          chat_history: historyForApi
        }),
        signal: streamAbort.signal
      });

      removeTypingIndicator();

      if (!response.ok) throw new Error('HTTP ' + response.status);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        let currentEvent = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim();
          } else if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (currentEvent === 'token') {
              fullReply += data;
              if (botMsgIndex === -1) {
                chatHistory.push({ role: 'bot', content: fullReply, time: formatTime() });
                botMsgIndex = chatHistory.length - 1;
              } else {
                chatHistory[botMsgIndex].content = fullReply;
              }
              renderMessages();
              const lastMsg = messagesEl.querySelector('.acw-msg-bot:last-child .acw-msg-bubble');
              if (lastMsg) lastMsg.textContent = fullReply;
            } else if (currentEvent === 'done') {
              try {
                meta = JSON.parse(data);
                if (botMsgIndex >= 0 && meta.reply) {
                  chatHistory[botMsgIndex].content = meta.reply;
                  chatHistory[botMsgIndex].status = meta.status;
                  chatHistory[botMsgIndex].sentiment = meta.sentiment_label;
                  chatHistory[botMsgIndex].intent = meta.intent;
                }
              } catch(e) {}
            }
            currentEvent = '';
          }
        }
      }

      if (botMsgIndex >= 0) {
        renderMessages();
      } else if (fullReply) {
        chatHistory.push({ role: 'bot', content: fullReply, time: formatTime(), status: meta.status });
        renderMessages();
      }

      if (!isOpen && meta.status !== 'human') {
        updateBadge(true);
      }

      saveHistory();
    } catch(e) {
      removeTypingIndicator();
      if (e.name === 'AbortError') {
        chatHistory.push({ role: 'bot', content: '已取消', time: formatTime() });
      } else {
        chatHistory.push({ role: 'bot', content: '网络错误，请稍后重试', time: formatTime() });
      }
      renderMessages();
      saveHistory();
    }

    isStreaming = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }

  function togglePanel() {
    isOpen = !isOpen;
    panelEl.classList.toggle('acw-open', isOpen);
    updateBadge(false);
    if (isOpen) {
      inputEl.focus();
      messagesEl.scrollTop = messagesEl.scrollHeight;
    } else if (isStreaming && streamAbort) {
      streamAbort.abort();
    }
  }

  function buildWidget() {
    const root = document.getElementById(ROOT_ID);
    root.innerHTML = '<div class="acw-panel"><div class="acw-header"><div class="acw-header-avatar">&#x1F916;</div><div class="acw-header-info"><div class="acw-header-title">&#x667A;&#x80FD;&#x5BA2;&#x670D;</div><div class="acw-header-status">&#x5728;&#x7EBF;</div></div><button class="acw-close" id="acw-close">&#x2715;</button></div><div class="acw-messages" id="acw-messages"></div><div class="acw-input-area"><textarea class="acw-input" id="acw-input" placeholder="&#x8F93;&#x5165;&#x60A8;&#x7684;&#x95EE;&#x9898;..." rows="1"></textarea><button class="acw-send" id="acw-send"><svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg></button></div></div><button class="acw-fab" id="acw-fab"><svg viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z"/></svg><div class="acw-badge" id="acw-badge"></div></button>';

    panelEl = root.querySelector('.acw-panel');
    messagesEl = root.querySelector('#acw-messages');
    inputEl = root.querySelector('#acw-input');
    sendBtn = root.querySelector('#acw-send');
    fabEl = root.querySelector('#acw-fab');
    badgeEl = root.querySelector('#acw-badge');

    root.querySelector('#acw-close').addEventListener('click', togglePanel);
    fabEl.addEventListener('click', togglePanel);
    sendBtn.addEventListener('click', sendMessage);

    inputEl.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    inputEl.addEventListener('input', function() {
      inputEl.style.height = 'auto';
      inputEl.style.height = Math.min(inputEl.scrollHeight, 80) + 'px';
    });

    renderMessages();
  }

  function init() {
    injectStyles();
    createRoot();
    buildWidget();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
