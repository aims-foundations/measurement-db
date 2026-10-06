export type SiteSearchResult = {
  id: string;
  href: string;
  title: string;
  summary: string;
  section: string;
  kind: string;
};

export type SiteSearchResponse = {
  results: readonly SiteSearchResult[];
  total: number;
};
