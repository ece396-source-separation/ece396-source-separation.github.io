---
title: "Blog"
author_profile: true
layout: archive
---

<ul class="taxonomy__index">
    <!-- <a href="#{{ site.categories }}"> -->
        {% for post in site.categories.Blog %}
        <li><span>{{ post.date | date_to_string }}</span> &nbsp; <a href="{{ post.url }}">{{ post.title }}</a></li>
       {% endfor %}
    
</ul>