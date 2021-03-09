## Proposal Purpose

> One sentence (maximum of 255 characters including spaces)

This proposal seeks to meet the rising demands for high-performance Python data visualization, to improve stability and community in PyQtGraph, and to attend to the experimental neuroscience issues as its used by ACQ4, in particular.

## Proposal Summary / Scope of Work
> A short summary of the application (maximum of 500 words)


Originally built to support experimental neuroscience, in the last decade, PyQtGraph has grown to be the 2nd most popular visualization library in the scientific Python ecosystem, supporting numerous real-time applications in bio-medical research and beyond. While its many contributors have been motivated to add features and fix critical bugs, recent efforts by dedicated maintainers have begun the process of shaping PyQtGraph into a stable, reliable cornerstone of the scientific community. Open pull requests are down nearly 90%. Continuous integration is catching regressions and bugs before they're merged. Predictable versioning and support have given our users a future they can rely on.

We seek to continue these efforts and expand on them, meeting the ever-increasing demands to visualize data quickly and beautifully, as well as make development on these applications intuitive, pythonic and stable. Much of this will involve tasks often deemed tedious, tasks that are often hard to organize amongst a team of volunteers. We need to expand our cloud infrastructure, to better represent the intense applications in which our library is used. We want to dedicate developer time to responding quickly to our user demands. 

[TODO paragraph on development work to be done, summarizing ideas so far]

We will expand our cloud infrastructure to better support the community needs. An oft-overlooked need of a library - especially one expected to provide high-performance features - is to have continuous benchmarking conducted on dedicated hardware. We aim to address this by expanding on the free-tier and volunteer-contributed testing currently used, into relatively expensive paid cloud servers. In this way, maintaining and improving the performance profile of PyQtGraph will become integrated with all development efforts.

Of less broad impact, but related in origin, tied together by community, and critical to its limited users, ACQ4 has sought over its long life to meet all the needs of the field of experimental neuropysiology. When existing Python data visualization was too slow, it spawned PyQtGraph. When Python bindings for specialized hardware didn't exist, it made those. It has partnered with manufacturers, labs and institutions across the world to create an open source omnibus of experimental neuroanatomy tools. Bringing this application through into modern Python has required months of tedious, unglamorous work. Now we'd like to give our users much needed improvements in UX and address larger architectural issues. Further, test infrastructure is impossible on commodity hardware, as even basic lab setups are prohibitively expensive. We would tap our manufacturer connections to help establish a community-accessible test rig on which to run continuous integration or other development tasks.


## Landscape Analysis

> Briefly describe the other software tools (either proprietary or open
> source) that the audience for this proposal is primarily using. How
> do the software projects in this proposal compare to these other
> tools in terms of size of user base, usage, and maturity? ​How do
> existing tools and the project(s) in this proposal interact? (maximum
> of 250 words)

PyQtGraph is the second-most widely used visualization library in Python and is a mature, established library (11+ years old) with nearly 200 individual code contributors and half a dozen currently active maintainers. Among the many options for visualization, PyQtGraph stands out as the preferred choice for real-time, interactive, high-performance applications. It also offers numerous features for building scientific applications beyond the visualization itself. To handle the demands of experiments on living tissues, scientists need responsive, performant data visualization, which PyQtGraph provides.

ACQ4 is the only complete, general solution to experimental neurophysiology. It supports all major hardware and most uncommon ones. It supports all standard experimental methodologies (electrophysiology, imaging, and photostimulation) and many novel ones. This is in contrast to a fragmented landscape of single-vendor, single-purpose, or single-institution software, none of which handle automated experiments between diverse hardware. Without ACQ4, most labs need to cobble together custom scripts per experiment or run everything manually.

## Value to Biomedical Users

> Briefly described the expected value the proposed scope of work will deliver to the biomedical research community (maximum of 250 words)

Biomedical science is only going to increase in its needs, in its demands on computers and software. Data is only going to grow larger. The need to make rapid responses to that steam of data must be met by a concerted effort to make our libraries faster, more performant and ever stable. PyQtGraph is already rising to meet those needs, but we'll increase the scope of what is possible by advancing that progress.

ACQ4 is used by labs the world over, backed by key manufacturers in the industry, and integrated into many institutions. Making the project more welcoming to newcomers and keeping this application and its ecosystem healthy allows for growth and advancement in all of neuroscience. 
