---
layout: default
title: All references
section_label: Reading List
permalink: /readinglist/
---

References cited across blog posts. External links open the original source.

{% assign items = site.readinglist | sort: "date" | reverse %}

## Papers

<div class="table-wrap">
<table>
<thead>
<tr><th>Title</th><th>Authors</th><th>Year</th></tr>
</thead>
<tbody>
{% for item in items %}
{% if item.title and item.source and item.kind == "paper" %}
<tr>
  <td><a href="{{ item.source }}" target="_blank" rel="noopener noreferrer">{{ item.title }}</a></td>
  <td>{% if item.authors %}{{ item.authors }}{% else %}—{% endif %}</td>
  <td><em>{{ item.date | date: "%Y" }}</em></td>
</tr>
{% endif %}
{% endfor %}
</tbody>
</table>
</div>

## Tools, docs & other

<div class="table-wrap">
<table>
<thead>
<tr><th>Title</th><th>Authors</th><th>Year</th></tr>
</thead>
<tbody>
{% for item in items %}
{% if item.title and item.source and item.kind == "resource" %}
<tr>
  <td><a href="{{ item.source }}" target="_blank" rel="noopener noreferrer">{{ item.title }}</a></td>
  <td>{% if item.authors %}{{ item.authors }}{% else %}—{% endif %}</td>
  <td><em>{{ item.date | date: "%Y" }}</em></td>
</tr>
{% endif %}
{% endfor %}
</tbody>
</table>
</div>
