if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
    document.querySelector("html").setAttribute("data-bs-theme", "dark");
}

const buttonGenerate = document.querySelector("button");
const spinner = document.querySelector(".spinner-border");
const promptInput = document.querySelector("textarea");
const generatedContainer = document.querySelector("#generated");
const alert = document.querySelector(".alert");

function disableInputs(disabled) {
    if (disabled) {
        spinner.classList.remove("d-none");
        buttonGenerate.setAttribute("disabled", "");
    } else {
        spinner.classList.add("d-none");
        buttonGenerate.removeAttribute("disabled");
    }
}

function renderGeneratedText(response) {
    const { prompt, generated } = JSON.parse(response);
    generatedContainer.innerHTML = '';

    const promptContainer = document.createElement("span");
    promptContainer.classList.add("text-secondary");
    promptContainer.innerText = prompt;
    generatedContainer.appendChild(promptContainer);

    const generatedWithoutPrompt = generated.slice(prompt.length);
    generatedContainer.appendChild(document.createTextNode(generatedWithoutPrompt));
}

async function generate(prompt, maxGenTokens, temp, topP) {
    const body = JSON.stringify({
        prompt,
        maxGenTokens,
        temp,
        topP,
    });

    const response = await fetch("/generate", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body,
    });

    const generated = await response.text();
    if (!response.ok) {
        throw new Error(`got ${response.status} from endpoint: ${generated}`);
    }

    renderGeneratedText(generated);
}

buttonGenerate.addEventListener("click", async () => {
    alert.classList.add("d-none");
    disableInputs(true);

    try {
        const maxGenTokens = parseInt(document.querySelector("#max-gen-tokens").value);
        const temp = parseFloat(document.querySelector("#temperature").value);
        const topP = parseFloat(document.querySelector("#top-p").value);
        await generate(promptInput.value, maxGenTokens, temp, topP);
    } catch (e) {
        alert.innerText = `ERROR: ${e}`;
        alert.classList.remove("d-none");
    } finally {
        disableInputs(false);
    }
});

const eventSource = new EventSource("/listen");
eventSource.addEventListener("partial", (event) => {
    // in case another tab/client is generating
    disableInputs(true);
    renderGeneratedText(JSON.parse(event.data));
});

eventSource.addEventListener("complete", (event) => {
    disableInputs(false);
});