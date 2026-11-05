// Local, blocking bootstrap: establish the theme before the page is painted.
(function () {
  var saved;
  try { saved = localStorage.getItem('theme'); } catch (_) {}
  var theme = saved === 'light' || saved === 'dark' ? saved :
    (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  document.documentElement.classList.add(theme);
  document.documentElement.style.colorScheme = theme;
})();
