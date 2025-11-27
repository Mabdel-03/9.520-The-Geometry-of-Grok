import yaml
from jinja2 import Template

template = Template(r"""
\begin{table}[h]
\centering
\textbf{ {{ metadata.title }} } \\[6pt]

\begin{tabular}{ll|l}
\hline
Model & Optimizer/Hyperparameters & Epoches & Peak VC Ratio Epoch & Peak VC Ratio\\
\hline
{% for v in trials %}
{{ v.model }} & {{ v.hyper }} & {{ v.epoches }} & {{ v.peakVCratioEpoch}} & {{ v.peakVCratio }} \\
{% endfor %}
\hline
\end{tabular}

\caption{ {{ metadata.caption }} }
\end{table}
""")

def render_paper_table(paper_id):
    data = yaml.safe_load(open(f"{paper_id}.yaml"))
    print(template.render(**data))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["render_paper_table"])
    parser.add_argument("paper_id")
    args = parser.parse_args()
    if args.command == "render_paper_table":
        render_paper_table(args.paper_id)