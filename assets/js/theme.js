(function () {
  const key = "cv-theme";
  const btn = document.getElementById("theme-btn");

  function syncLabel() {
    if (!btn) return;
    const dark = document.documentElement.dataset.theme === "dark";
    btn.textContent = dark ? "Light" : "Dark";
    btn.title = dark ? "Switch to light theme" : "Switch to dark theme";
  }

  btn?.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme === "dark";
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem(key, next);
    syncLabel();
  });

  syncLabel();

  const sidebarKey = "cv-sidebar";
  const sidebarBtn = document.getElementById("sidebar-toggle");
  const cornellBody = document.querySelector(".cornell-body");

  function syncSidebar() {
    if (!sidebarBtn || !cornellBody) return;
    const collapsed = cornellBody.classList.contains("sidebar-collapsed");
    sidebarBtn.setAttribute("aria-expanded", collapsed ? "false" : "true");
    sidebarBtn.title = collapsed ? "Show sidebar" : "Hide sidebar";
    sidebarBtn.setAttribute("aria-label", sidebarBtn.title);
  }

  const isMobile = window.matchMedia("(max-width: 768px)").matches;
  const storedSidebar = localStorage.getItem(sidebarKey);
  if (storedSidebar === "collapsed" || (storedSidebar === null && isMobile)) {
    cornellBody?.classList.add("sidebar-collapsed");
  }

  sidebarBtn?.addEventListener("click", () => {
    cornellBody?.classList.toggle("sidebar-collapsed");
    localStorage.setItem(
      sidebarKey,
      cornellBody?.classList.contains("sidebar-collapsed") ? "collapsed" : "open"
    );
    syncSidebar();
  });

  syncSidebar();

  document.querySelectorAll(".nav-group-link").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.stopPropagation();
    });
  });

  const resumeNav = document.getElementById("nav-resume");
  const blogNav = document.getElementById("nav-blog");
  const readingNav = document.getElementById("nav-readinglist");
  if (resumeNav instanceof HTMLDetailsElement && blogNav instanceof HTMLDetailsElement) {
    const path = window.location.pathname;
    const onBlog = path.startsWith("/blog/");
    const onReading = path.startsWith("/readinglist/");
    resumeNav.open = !onBlog && !onReading;
    blogNav.open = onBlog;
    if (readingNav instanceof HTMLDetailsElement) readingNav.open = onReading;
  }
})();
