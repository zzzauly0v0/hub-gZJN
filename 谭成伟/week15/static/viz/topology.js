/* topology.js — 规划拓扑可视化（SVG，零依赖，环形辐射布局，旅行主题） */
class TopoViz {
  constructor(host) {
    this.host = host;
    this.host.innerHTML = '';
    this.svgNS = 'http://www.w3.org/2000/svg';
    this.subs = {};
    this.order = [];
    this._clickCb = null;
    this.W = 380; this.H = 480;
    this.cx = this.W / 2;
    this.cy = 240;
    this.radius = 150;
    this.svg = this._svg();
    this.host.appendChild(this.svg);
    const defs = document.createElementNS(this.svgNS, 'defs');
    defs.innerHTML = `
      <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
        <feGaussianBlur stdDeviation="3.5" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
      <radialGradient id="mainGrad" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#5eead4"/>
        <stop offset="100%" stop-color="#0f766e"/>
      </radialGradient>
      <radialGradient id="subGrad" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#34d399"/>
        <stop offset="100%" stop-color="#065f46"/>
      </radialGradient>`;
    this.svg.appendChild(defs);
  }

  _svg() {
    const s = document.createElementNS(this.svgNS, 'svg');
    s.setAttribute('viewBox', `0 0 ${this.W} ${this.H}`);
    s.setAttribute('width', '100%');
    s.setAttribute('style',
      'background:radial-gradient(circle at 50% 50%, #0f2a26 0%, #07120f 80%);border-radius:8px');
    return s;
  }

  _node(x, y, r, fill, label, id, glowColor, fontSize = 8.5) {
    const g = document.createElementNS(this.svgNS, 'g');
    g.style.cursor = 'pointer';
    const c = document.createElementNS(this.svgNS, 'circle');
    c.setAttribute('cx', x); c.setAttribute('cy', y); c.setAttribute('r', r);
    c.setAttribute('fill', fill);
    c.setAttribute('stroke', glowColor || '#2dd4bf');
    c.setAttribute('stroke-width', '2');
    c.setAttribute('filter', 'url(#glow)');
    c.style.transition = 'all .3s';
    const t = document.createElementNS(this.svgNS, 'text');
    t.setAttribute('x', x); t.setAttribute('y', y + r + 13);
    t.setAttribute('text-anchor', 'middle'); t.setAttribute('font-size', fontSize);
    t.setAttribute('fill', '#8acfbf');
    t.textContent = label;
    g.appendChild(c); g.appendChild(t);
    if (id) g.addEventListener('click', () => this._clickCb && this._clickCb(id));
    this.svg.appendChild(g);
    return { g, c, t };
  }

  _edge(x1, y1, x2, y2, color) {
    const ln = document.createElementNS(this.svgNS, 'line');
    ln.setAttribute('x1', x1); ln.setAttribute('y1', y1);
    ln.setAttribute('x2', x2); ln.setAttribute('y2', y2);
    ln.setAttribute('stroke', color || '#1f5a4f');
    ln.setAttribute('stroke-width', '1.5');
    ln.setAttribute('stroke-dasharray', '6 6');
    const anim = document.createElementNS(this.svgNS, 'animate');
    anim.setAttribute('attributeName', 'stroke-dashoffset');
    anim.setAttribute('from', '24');
    anim.setAttribute('to', '0');
    anim.setAttribute('dur', '0.8s');
    anim.setAttribute('repeatCount', 'indefinite');
    ln.appendChild(anim);
    this.svg.appendChild(ln);
    return ln;
  }

  setMain() {
    const o = this._node(this.cx, this.cy, 20, 'url(#mainGrad)', '主 agent', 'main', '#2dd4bf', 9);
    this.subs['main'] = { ...o, x: this.cx, y: this.cy, status: 'idle' };
  }

  addSubagent(id, topic) {
    const i = this.order.length;
    const n = this.order.length + 1;
    const angle = (-Math.PI / 2) + (i * 2 * Math.PI / Math.max(n, 3));
    const x = this.cx + this.radius * Math.cos(angle);
    const y = this.cy + this.radius * Math.sin(angle);
    const o = this._node(x, y, 14, 'url(#subGrad)',
      topic.length > 8 ? topic.slice(0, 8) + '…' : topic, id, '#34d399');
    const dx = x - this.cx, dy = y - this.cy;
    const dist = Math.hypot(dx, dy) || 1;
    this._edge(this.cx + dx / dist * 20, this.cy + dy / dist * 20,
               x - dx / dist * 14, y - dy / dist * 14);
    this.subs[id] = { ...o, x, y, status: 'idle', topic };
    this.order.push(id);
  }

  markRunning(id) {
    const s = this.subs[id]; if (!s) return;
    s.status = 'running';
    s.c.setAttribute('stroke', '#fbbf24'); s.c.setAttribute('stroke-width', '3.5');
    s.c.setAttribute('fill', '#3a2e00');
    if (!s._pulse) {
      s._pulse = setInterval(() => {
        if (s.status !== 'running') { clearInterval(s._pulse); s._pulse = null; return; }
        s.c.setAttribute('r', s.c.getAttribute('r') === '16' ? '13' : '16');
      }, 450);
    }
  }

  markDone(id) {
    const s = this.subs[id]; if (!s) return;
    s.status = 'done';
    if (s._pulse) { clearInterval(s._pulse); s._pulse = null; }
    s.c.setAttribute('fill', '#0d3d2a'); s.c.setAttribute('stroke', '#34d399');
    s.c.setAttribute('stroke-width', '2.5'); s.c.setAttribute('r', '14');
  }

  reset() {
    Object.values(this.subs).forEach(s => { if (s._pulse) clearInterval(s._pulse); });
    this.host.innerHTML = '';
    this.subs = {}; this.order = [];
  }

  onClick(cb) { this._clickCb = cb; }
}
