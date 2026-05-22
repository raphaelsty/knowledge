// Mirror of sources/utils/client.py::_hostname_source_key.
//
// Website fetchers bucket docs by the URL's hostname so each site
// ends up with its own filter chip (e.g. `mixedbread.com`) instead
// of a generic "blog" bucket.
export function hostnameSourceKey(url) {
  try {
    const host = new URL(url).hostname.toLowerCase();
    if (!host) return "";
    return host.startsWith("www.") ? host.slice(4) : host;
  } catch {
    return "";
  }
}
