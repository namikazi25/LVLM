"""Web Search Module for MMFakeBench.

This module contains the WebSearcher class that performs web searches using
multiple search engines (Brave Search and DuckDuckGo) to gather evidence
for fact-checking claims.
"""

import os
import logging
import requests
import time
from typing import Dict, Any, List, Optional, Union
from urllib.parse import quote_plus

from core.base import BasePipelineModule


class WebSearcher(BasePipelineModule):
    """Pipeline module for web search functionality.
    
    This module performs web searches using multiple search engines to gather
    evidence and supporting information for fact-checking claims.
    """
    
    def __init__(self, 
                 name: str = "web_searcher",
                 brave_api_key: Optional[str] = None,
                 max_results: int = 10,
                 timeout: int = 30,
                 retry_attempts: int = 3,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 **kwargs):
        """Initialize the web searcher.
        
        Args:
            name: Name of the module
            brave_api_key: API key for Brave Search
            max_results: Maximum number of search results to return
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retry attempts in seconds
            **kwargs: Additional configuration parameters
        """
        config = {
            'brave_api_key': brave_api_key,
            'max_results': max_results,
            'timeout': timeout,
            'retry_attempts': retry_attempts,
            'max_retries': max_retries,
            'retry_delay': retry_delay,
            **kwargs
        }
        super().__init__(name=name, config=config)
        self.brave_api_key = brave_api_key or os.getenv('BRAVE_API_KEY')
        self.max_results = max_results
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        
        # Search engine endpoints
        self.brave_endpoint = "https://api.search.brave.com/res/v1/web/search"
        self.duckduckgo_endpoint = "https://api.duckduckgo.com/"
    
    def initialize(self) -> None:
        """Initialize the module with its configuration."""
        try:
            if not self.brave_api_key:
                self.logger.warning("No Brave API key provided. Brave Search will be disabled.")
            
            self.logger.info(f"WebSearcher initialized with max_results={self.max_results}")
        except Exception as e:
            self.logger.error(f"Failed to initialize WebSearcher: {e}")
            raise
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data format.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        required_fields = ['query']
        
        for field in required_fields:
            if field not in data:
                self.logger.error(f"Missing required field: {field}")
                return False
        
        if not isinstance(data['query'], str) or not data['query'].strip():
            self.logger.error("Query must be a non-empty string")
            return False
        
        return True
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get the expected output schema for this module.
        
        Returns:
            Dictionary describing the output schema
        """
        return {
            'query': 'str',
            'search_results': {
                'brave_results': 'List[Dict]',
                'duckduckgo_results': 'List[Dict]',
                'combined_results': 'List[Dict]'
            },
            'search_metadata': {
                'total_results': 'int',
                'search_engines_used': 'List[str]',
                'search_time': 'float'
            },
            'module_status': 'str'
        }
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data through web search.
        
        Args:
            data: Input data dictionary containing 'query'
            
        Returns:
            Processed data dictionary with search results
        """
        start_time = time.time()
        query = data['query'].strip()
        
        self.logger.info(f"Performing web search for query: {query}")
        
        # Initialize results
        brave_results = []
        duckduckgo_results = []
        engines_used = []
        
        # Search with Brave if API key is available
        if self.brave_api_key:
            try:
                brave_results = self._search_brave(query)
                engines_used.append('brave')
                self.logger.info(f"Brave Search returned {len(brave_results)} results")
            except Exception as e:
                self.logger.error(f"Brave Search failed: {e}")
        
        # Search with DuckDuckGo
        try:
            duckduckgo_results = self._search_duckduckgo(query)
            engines_used.append('duckduckgo')
            self.logger.info(f"DuckDuckGo Search returned {len(duckduckgo_results)} results")
        except Exception as e:
            self.logger.error(f"DuckDuckGo Search failed: {e}")
        
        # Combine and deduplicate results
        combined_results = self._combine_results(brave_results, duckduckgo_results)
        
        search_time = time.time() - start_time
        
        return {
            **data,
            'search_results': {
                'brave_results': brave_results,
                'duckduckgo_results': duckduckgo_results,
                'combined_results': combined_results
            },
            'search_metadata': {
                'total_results': len(combined_results),
                'search_engines_used': engines_used,
                'search_time': search_time
            }
        }
    
    def _search_brave(self, query: str) -> List[Dict[str, Any]]:
        """Search using Brave Search API.
        
        Args:
            query: Search query string
            
        Returns:
            List of search result dictionaries
        """
        headers = {
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'X-Subscription-Token': self.brave_api_key
        }
        
        params = {
            'q': query,
            'count': min(self.max_results, 20),  # Brave API limit
            'search_lang': 'en',
            'country': 'US',
            'safesearch': 'moderate',
            'freshness': 'pw'  # Past week for recent information
        }
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(
                    self.brave_endpoint,
                    headers=headers,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                results = []
                
                if 'web' in data and 'results' in data['web']:
                    for item in data['web']['results']:
                        results.append({
                            'title': item.get('title', ''),
                            'url': item.get('url', ''),
                            'snippet': item.get('description', ''),
                            'source': 'brave',
                            'published_date': item.get('age', '')
                        })
                
                return results[:self.max_results]
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Brave Search attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        return []
    
    def _search_duckduckgo(self, query: str) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo Instant Answer API.
        
        Args:
            query: Search query string
            
        Returns:
            List of search result dictionaries
        """
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.get(
                    self.duckduckgo_endpoint,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                results = []
                
                # Process instant answer
                if data.get('Abstract'):
                    results.append({
                        'title': data.get('Heading', 'DuckDuckGo Instant Answer'),
                        'url': data.get('AbstractURL', ''),
                        'snippet': data.get('Abstract', ''),
                        'source': 'duckduckgo_instant',
                        'published_date': ''
                    })
                
                # Process related topics
                for topic in data.get('RelatedTopics', [])[:self.max_results//2]:
                    if isinstance(topic, dict) and 'Text' in topic:
                        results.append({
                            'title': topic.get('Text', '')[:100] + '...',
                            'url': topic.get('FirstURL', ''),
                            'snippet': topic.get('Text', ''),
                            'source': 'duckduckgo_related',
                            'published_date': ''
                        })
                
                return results[:self.max_results]
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"DuckDuckGo Search attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        
        return []
    
    def _combine_results(self, brave_results: List[Dict], duckduckgo_results: List[Dict]) -> List[Dict]:
        """Combine and deduplicate search results from multiple engines.
        
        Args:
            brave_results: Results from Brave Search
            duckduckgo_results: Results from DuckDuckGo
            
        Returns:
            Combined and deduplicated list of results
        """
        combined = []
        seen_urls = set()
        
        # Add Brave results first (typically higher quality)
        for result in brave_results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                combined.append(result)
        
        # Add DuckDuckGo results, avoiding duplicates
        for result in duckduckgo_results:
            url = result.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                combined.append(result)
        
        # Sort by relevance (Brave first, then DuckDuckGo)
        combined.sort(key=lambda x: (x['source'] != 'brave', x['source'] != 'duckduckgo_instant'))
        
        return combined[:self.max_results]
    
    def search_claim(self, claim: str, additional_terms: Optional[List[str]] = None) -> Dict[str, Any]:
        """Convenience method to search for a specific claim.
        
        Args:
            claim: The claim to search for
            additional_terms: Additional search terms to include
            
        Returns:
            Search results dictionary
        """
        query = claim
        if additional_terms:
            query += " " + " ".join(additional_terms)
        
        return self.process({'query': query})