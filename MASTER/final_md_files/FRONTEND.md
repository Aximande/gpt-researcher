

# File: pdf_styles.css

body {
    font-family: 'Libre Baskerville', serif;
    font-size: 12pt; /* standard size for academic papers */
    line-height: 1.6; /* for readability */
    color: #333; /* softer on the eyes than black */
    background-color: #fff; /* white background */
    margin: 0;
    padding: 0;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Libre Baskerville', serif;
    color: #000; /* darker than the body text */
    margin-top: 1em; /* space above headers */
}

h1 {
    font-size: 2em; /* make h1 twice the size of the body text */
}

h2 {
    font-size: 1.5em;
}

/* Add some space between paragraphs */
p {
    margin-bottom: 1em;
}

/* Style for blockquotes, often used in academic papers */
blockquote {
    font-style: italic;
    margin: 1em 0;
    padding: 1em;
    background-color: #f9f9f9; /* a light grey background */
}

/* You might want to style tables, figures, etc. too */
table {
    border-collapse: collapse;
    width: 100%;
}

table, th, td {
    border: 1px solid #ddd;
    text-align: left;
    padding: 8px;
}

th {
    background-color: #f2f2f2;
    color: black;
}

# File: index.html

<!DOCTYPE html>
<html lang="en">

<head>
    <title>GPT Researcher</title>
    <meta name="description" content="A research assistant powered by GPT-4">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="icon" href="./static/favicon.ico">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="/site/styles.css"/>
    <style>
        .avatar {
            width: 60px;
            height: 60px;
            border-radius: 50%;
        }

        .agent-name {
            text-align: center;
        }

        .agent-item {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .agent-choices {
            display: none;
        }

        .btn-show {
            display: none;
        }
    </style>
</head>

<body>

<section class="landing">
    <div class="max-w-5xl mx-auto text-center">
        <h1 class="text-4xl font-extrabold mx-auto lg:text-7xl">
            Say Goodbye to <br>
            <span
                    style="background-image:linear-gradient(to right, #9867F0, #ED4E50); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Hours
                    of Research</span>
        </h1>
        <p class="max-w-5xl mx-auto text-gray-600 mt-8" style="font-size:20px">
            Say Hello to GPT Researcher, your AI mate for rapid insights and comprehensive research. GPT Researcher
            takes care of everything from accurate source gathering to organization of research results - all in one
            platform designed to make your research process a breeze.
        </p>
        <a href="#form" class="btn btn-primary">Get Started</a>
    </div>
</section>

<main class="container" id="form">
    <div class="agent-item"><img src="/static/defaultAgentAvatar.JPG" class="avatar"
                                                alt="Auto Agent"></div>
    <form method="POST" class="mt-3" onsubmit="GPTResearcher.startResearch(); return false;">
        <div class="form-group">
            <label for="task" class="agent-question">What would you like me to research next?</label>
            <input type="text" id="task" name="task" class="form-control" required>
            <input type="radio" name="agent" id="autoAgent" value="Auto Agent" checked hidden>
        </div>
        <div class="form-group">
            <div class="row">


            </div>
            <button type="button" id="btnShowAuto" class="btn btn-secondary mt-3 btn-show">Auto Agent</button>
        </div>
        <div class="form-group">
            <label for="report_type" class="agent-question">What type of report would you like me to generate?</label>
            <select name="report_type" class="form-control" required>
                <option value="research_report">Summary - Short and fast (~2 min)</option>
                <option value="detailed_report">Detailed - In depth and longer (~5 min)</option>
                <option value="resource_report">Resource Report</option>
            </select>
        </div>
        <div class="form-group">
</div>
        <input type="submit" value="Research" class="btn btn-primary button-padding">
    </form>

    <div class="margin-div">
        <h2>Agent Output</h2>
        <p class="mt-2 text-left" style="font-size: 0.8rem;">An agent tailored specifically to your task
                        will be generated to provide the most precise and relevant research results.</p>
        <div id="output"></div>
    </div>
    <div class="margin-div">
        <h2>Research Report</h2>
        <div id="reportContainer"></div>
        <div id="reportActions">
            <div class="alert alert-info" role="alert" id="status"></div>
            <a id="copyToClipboard" onclick="GPTResearcher.copyToClipboard()" class="btn btn-secondary mt-3" style="margin-right: 10px;">Copy to clipboard</a>
            <a id="downloadLink" href="#" class="btn btn-secondary mt-3" style="margin-right: 10px;" target="_blank">Download as PDF</a>
            <a id="downloadLinkWord" href="#" class="btn btn-secondary mt-3" target="_blank">Download as Docx</a>
        </div>
    </div>
</main>

<footer>
    <p>GPT Researcher &copy; 2024 | <a target="_blank" href="https://github.com/assafelovic/gpt-researcher">GitHub
        Page</a></p>
</footer>

<script src="https://cdnjs.cloudflare.com/ajax/libs/showdown/1.9.1/showdown.min.js"></script>
<script src="/site/scripts.js"></script>
<script>
    // const btnChoose = document.getElementById('btnChoose');
    const btnShowAuto = document.getElementById('btnShowAuto');
    const autoAgentDiv = document.getElementById('autoAgentDiv');
    const agentChoices = document.getElementsByClassName('agent-choices');

    /**
    btnChoose.addEventListener('click', function () {
        btnShowAuto.style.display = 'inline-block';
        btnChoose.style.display = 'none';
        autoAgentDiv.style.display = 'none';
        agentChoices[0].style.display = 'flex';
    });
    **/

    btnShowAuto.addEventListener('click', function () {
        btnShowAuto.style.display = 'none';
        btnChoose.style.display = 'inline-block';
        autoAgentDiv.style.display = 'flex';
        agentChoices[0].style.display = 'none';
    });
</script>
</body>

</html>


# File: styles.css

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

body {
    font-family: 'Montserrat', sans-serif;
    color: #fff;
    line-height: 1.6;
    background-size: 200% 200%;
    background-image: linear-gradient(45deg, #151A2D, #2D284D, #151A2D);
    animation: gradientBG 10s ease infinite;
}

.landing {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    text-align: center;
}

.landing h1 {
    font-size: 3.5rem;
    font-weight: 700;
    margin-bottom: 2rem;
}

.landing p {
    font-size: 1.5rem;
    font-weight: 400;
    max-width: 500px;
    margin: auto;
    margin-bottom: 2rem;
}

.container {
    max-width: 900px;
    margin: auto;
    padding: 20px;
    background-color: rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    box-shadow: 0px 10px 25px rgba(0, 0, 0, 0.1);
    transition: all .3s ease-in-out;
    margin-bottom: 180px;
}

.container:hover {
    transform: scale(1.01);
    box-shadow: 0px 15px 30px rgba(0, 0, 0, 0.2);
}

input, select, #output, #reportContainer {
    background-color: rgba(255,255,255,0.1);
    border: none;
    color: #fff;
    transition: all .3s ease-in-out;
}

input:hover, input:focus, select:hover, select:focus {
    background-color: #dfe4ea;
    border: 1px solid rgba(255, 255, 255, 0.5);
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease-in-out;
}

.btn-primary {
    background: linear-gradient(to right, #0062cc, #007bff);
    border: none;
    transition: all .3s ease-in-out;
}

.btn-secondary {
    background: linear-gradient(to right, #6c757d, #6c757d);
    border: none;
    transition: all .3s ease-in-out;
}

.btn:hover {
    opacity: 0.8;
    transform: scale(1.1);
    box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.3);
}

.agent_question {
    font-size: 1.2rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
}

footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background: linear-gradient(to right, #151A2D, #111827);
    color: white;
    text-align: center;
    padding: 10px 0;
}

.margin-div {
    margin-top: 20px;
    margin-bottom: 20px;
    padding: 10px;
}

.agent_response {
    background-color: #747d8c;
    margin: 10px;
    padding: 10px;
    border-radius: 12px;
}

#output {
    height: 300px;
    font-family: 'Times New Roman', Times, , "Courier New", serif;
    overflow: auto;
    padding: 10px;
    margin-bottom: 10px;
    margin-top: 10px;
}

#reportContainer {
    background-color: rgba(255,255,255,0.1);
    border: none;
    color: #fff;
    transition: all .3s ease-in-out;
    padding: 10px;
    border-radius: 12px;
}


# File: scripts.js

const GPTResearcher = (() => {
    const init = () => {
      // Not sure, but I think it would be better to add event handlers here instead of in the HTML
      //document.getElementById("startResearch").addEventListener("click", startResearch);
      document.getElementById("copyToClipboard").addEventListener("click", copyToClipboard);

      updateState("initial");
    }

    const startResearch = () => {
      document.getElementById("output").innerHTML = "";
      document.getElementById("reportContainer").innerHTML = "";
      updateState("in_progress")
  
      addAgentResponse({ output: "🤔 Thinking about research questions for the task..." });
  
      listenToSockEvents();
    };
  
    const listenToSockEvents = () => {
      const { protocol, host, pathname } = window.location;
      const ws_uri = `${protocol === 'https:' ? 'wss:' : 'ws:'}//${host}${pathname}ws`;
      const converter = new showdown.Converter();
      const socket = new WebSocket(ws_uri);
  
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'logs') {
          addAgentResponse(data);
        } else if (data.type === 'report') {
          writeReport(data, converter);
        } else if (data.type === 'path') {
          updateState("finished")
          updateDownloadLink(data);
        }
      };
  
      socket.onopen = (event) => {
        const task = document.querySelector('input[name="task"]').value;
        const report_type = document.querySelector('select[name="report_type"]').value;
        const agent = document.querySelector('input[name="agent"]:checked').value;
  
        const requestData = {
          task: task,
          report_type: report_type,
          agent: agent,
        };
  
        socket.send(`start ${JSON.stringify(requestData)}`);
      };
    };
  
    const addAgentResponse = (data) => {
      const output = document.getElementById("output");
      output.innerHTML += '<div class="agent_response">' + data.output + '</div>';
      output.scrollTop = output.scrollHeight;
      output.style.display = "block";
      updateScroll();
    };
  
    const writeReport = (data, converter) => {
      const reportContainer = document.getElementById("reportContainer");
      const markdownOutput = converter.makeHtml(data.output);
      reportContainer.innerHTML += markdownOutput;
      updateScroll();
    };
  
    const updateDownloadLink = (data) => {
      const pdf_path = data.output.pdf;
      const docx_path = data.output.docx;
      document.getElementById("downloadLink").setAttribute("href", pdf_path);
      document.getElementById("downloadLinkWord").setAttribute("href", docx_path);
    };
  
    const updateScroll = () => {
      window.scrollTo(0, document.body.scrollHeight);
    };
  
    const copyToClipboard = () => {
      const textarea = document.createElement('textarea');
      textarea.id = 'temp_element';
      textarea.style.height = 0;
      document.body.appendChild(textarea);
      textarea.value = document.getElementById('reportContainer').innerText;
      const selector = document.querySelector('#temp_element');
      selector.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
    };

    const updateState = (state) => {
      var status = "";
      switch (state) {
        case "in_progress":
          status = "Research in progress..."
          setReportActionsStatus("disabled");
          break;
        case "finished":
          status = "Research finished!"
          setReportActionsStatus("enabled");
          break;
        case "error":
          status = "Research failed!"
          setReportActionsStatus("disabled");
          break;
        case "initial":
          status = ""
          setReportActionsStatus("hidden");
          break;
        default:
          setReportActionsStatus("disabled");
      }
      document.getElementById("status").innerHTML = status;
      if (document.getElementById("status").innerHTML == "") {
        document.getElementById("status").style.display = "none";
      } else {
        document.getElementById("status").style.display = "block";
      }
    }

    /**
     * Shows or hides the download and copy buttons
     * @param {str} status Kind of hacky. Takes "enabled", "disabled", or "hidden". "Hidden is same as disabled but also hides the div"
     */
    const setReportActionsStatus = (status) => {
      const reportActions = document.getElementById("reportActions");
      // Disable everything in reportActions until research is finished

      if (status == "enabled") {
        reportActions.querySelectorAll("a").forEach((link) => {
          link.classList.remove("disabled");
          link.removeAttribute('onclick');
          reportActions.style.display = "block";
        });
      } else {
        reportActions.querySelectorAll("a").forEach((link) => {
          link.classList.add("disabled");
          link.setAttribute('onclick', "return false;");
        });
        if (status == "hidden") {
          reportActions.style.display = "none";
        }
      }
    }

    document.addEventListener("DOMContentLoaded", init);
    return {
      startResearch,
      copyToClipboard,
    };
  })();