#!/usr/bin/env python
"""Simple tools to query github.com and gather stats about issues.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function

import json
import re
import sys

from datetime import datetime, timedelta
from urllib import urlopen

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

ISO8601 = "%Y-%m-%dT%H:%M:%SZ"
PER_PAGE = 100

element_pat = re.compile(r'<(.+?)>')
rel_pat = re.compile(r'rel=[\'"](\w+)[\'"]')

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

def parse_link_header(headers):
    link_s = headers.get('link', '')
    urls = element_pat.findall(link_s)
    rels = rel_pat.findall(link_s)
    d = {}
    for rel,url in zip(rels, urls):
        d[rel] = url
    return d

def get_paged_request(url):
    """get a full list, handling APIv3's paging"""
    results = []
    while url:
        print("fetching %s" % url, file=sys.stderr)
        f = urlopen(url)
        results.extend(json.load(f))
        links = parse_link_header(f.headers)
        url = links.get('next')
    return results

def get_issues(project="nipy/nitime", state="closed", pulls=False):
    """Get a list of the issues from the Github API."""
    which = 'pulls' if pulls else 'issues'
    url = "https://api.github.com/repos/%s/%s?state=%s&per_page=%i" % (project, which, state, PER_PAGE)
    return get_paged_request(url)


def _parse_datetime(s):
    """Parse dates in the format returned by the Github API."""
    if s:
        return datetime.strptime(s, ISO8601)
    else:
        return datetime.fromtimestamp(0)


def issues2dict(issues):
    """Convert a list of issues to a dict, keyed by issue number."""
    idict = {}
    for i in issues:
        idict[i['number']] = i
    return idict


def is_pull_request(issue):
    """Return True if the given issue is a pull request."""
    return 'pull_request_url' in issue


def issues_closed_since(period=timedelta(days=730), project="nipy/nitime", pulls=False):
    """Get all issues closed since a particular point in time. period
can either be a datetime object, or a timedelta object. In the
latter case, it is used as a time before the present."""

    which = 'pulls' if pulls else 'issues'

    if isinstance(period, timedelta):
        period = datetime.now() - period
    url = "https://api.github.com/repos/%s/%s?state=closed&sort=updated&since=%s&per_page=%i" % (project, which, period.strftime(ISO8601), PER_PAGE)
    allclosed = get_paged_request(url)
    # allclosed = get_issues(project=project, state='closed', pulls=pulls, since=period)
    filtered = [i for i in allclosed if _parse_datetime(i['closed_at']) > period]
    return filtered


def sorted_by_field(issues, field='closed_at', reverse=False):
    """Return a list of issues sorted by closing date date."""
    return sorted(issues, key = lambda i:i[field], reverse=reverse)


def report(issues, show_urls=False):
    """Summary report about a list of issues, printing number and title.
    """
    # titles may have unicode in them, so we must encode everything below
    if show_urls:
        for i in issues:
            role = 'ghpull' if 'merged' in i else 'ghissue'
            print('* :%s:`%d`: %s' % (role, i['number'],
                                        i['title'].encode('utf-8')))
    else:
        for i in issues:
            print('* %d: %s' % (i['number'], i['title'].encode('utf-8')))

#-----------------------------------------------------------------------------
# Main script
#-----------------------------------------------------------------------------

if __name__ == "__main__":
    # Whether to add reST urls for all issues in printout.
    show_urls = True

    # By default, search one month back
    if len(sys.argv) > 1:
        days = int(sys.argv[1])
    else:
        days = 730

    # turn off to play interactively without redownloading, use %run -i
    if 1:
        issues = issues_closed_since(timedelta(days=days), pulls=False)
        pulls = issues_closed_since(timedelta(days=days), pulls=True)

    # For regular reports, it's nice to show them in reverse chronological order
    issues = sorted_by_field(issues, reverse=True)
    pulls = sorted_by_field(pulls, reverse=True)

    n_issues, n_pulls = map(len, (issues, pulls))
    n_total = n_issues + n_pulls

    # Print summary report we can directly include into release notes.
    print("GitHub stats for the last  %d days." % days)
    print("We closed a total of %d issues, %d pull requests and %d regular \n"
          "issues; this is the full list (generated with the script \n"
          "`tools/github_stats.py`):" % (n_total, n_pulls, n_issues))
    print()
    print('Pull Requests (%d):\n' % n_pulls)
    report(pulls, show_urls)
    print()
    print('Issues (%d):\n' % n_issues)
    report(issues, show_urls)
