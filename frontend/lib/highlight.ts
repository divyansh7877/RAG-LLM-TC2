/**
 * Simple text highlighting utility for search results
 */

export function highlightMatches(text: string, searchTerms: string | string[]): string {
  if (!text || !searchTerms) return text;
  
  const terms = Array.isArray(searchTerms) ? searchTerms : [searchTerms];
  if (terms.length === 0) return text;
  
  // Escape special regex characters and create pattern
  const escapedTerms = terms
    .filter(term => term.trim())
    .map(term => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    
  if (escapedTerms.length === 0) return text;
  
  const pattern = new RegExp(`(${escapedTerms.join('|')})`, 'gi');
  
  return text.replace(pattern, '<mark class="bg-yellow-200 rounded px-1">$1</mark>');
}

export function extractRelevantSnippet(
  text: string, 
  searchTerms: string | string[], 
  maxLength = 200
): string {
  if (!text || !searchTerms) return text.slice(0, maxLength) + (text.length > maxLength ? '...' : '');
  
  const terms = Array.isArray(searchTerms) ? searchTerms : [searchTerms];
  const firstTerm = terms[0]?.toLowerCase();
  
  if (!firstTerm) return text.slice(0, maxLength) + (text.length > maxLength ? '...' : '');
  
  const lowerText = text.toLowerCase();
  const matchIndex = lowerText.indexOf(firstTerm);
  
  if (matchIndex === -1) {
    return text.slice(0, maxLength) + (text.length > maxLength ? '...' : '');
  }
  
  // Calculate snippet boundaries
  const start = Math.max(0, matchIndex - Math.floor(maxLength / 2));
  const end = Math.min(text.length, start + maxLength);
  
  let snippet = text.slice(start, end);
  
  // Add ellipsis if needed
  if (start > 0) snippet = '...' + snippet;
  if (end < text.length) snippet = snippet + '...';
  
  return snippet;
}
