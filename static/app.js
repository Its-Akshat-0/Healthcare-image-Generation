const form = document.getElementById("query-form");
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const status = document.getElementById("status");
  const results = document.getElementById("results");
  const textDetails = document.getElementById("text-details");
  status.textContent = "Retrieving source passages and preparing the visual…";
  results.hidden = true;
  textDetails.open = false;
  try {
    const response = await fetch("/api/health-explain", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        disease: document.getElementById("disease").value,
        question: document.getElementById("question").value,
        generate_image: document.getElementById("make-image").checked
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Request failed");
    document.getElementById("result-title").textContent = data.disease;
    const matchNotice = document.getElementById("match-notice");
    matchNotice.textContent = data.match_notice || "";
    matchNotice.hidden = !data.match_notice;
    document.getElementById("summary").textContent = data.summary;
    const sections = document.getElementById("sections");
    sections.replaceChildren();
    for (const [heading, facts] of Object.entries(data.sections || {})) {
      const h = document.createElement("h3");
      h.textContent = heading;
      sections.appendChild(h);
      for (const fact of facts) {
        const p = document.createElement("p");
        p.className = "fact";
        p.textContent = fact;
        sections.appendChild(p);
      }
    }
    const imageArea = document.getElementById("image-area");
    imageArea.replaceChildren();
    const imageError = document.getElementById("image-error");
    imageError.textContent = "";
    imageError.hidden = true;
    if (data.image_url) {
      const h = document.createElement("h3"); h.textContent = "Educational poster";
      const img = document.createElement("img"); img.src = data.image_url + "?t=" + Date.now(); img.alt = "Poster with source-grounded health text and an illustrative image when available";
      imageArea.append(h, img);
      if (data.image_error) {
        imageError.textContent = "Colab image request failed: " + data.image_error;
        imageError.hidden = false;
      }
    }
    const sources = document.getElementById("sources");
    sources.replaceChildren();
    (data.sources || []).forEach(source => {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = source.url; a.textContent = source.name + " — " + source.document_id;
      a.target = "_blank"; a.rel = "noopener noreferrer";
      li.appendChild(a); sources.appendChild(li);
    });
    document.getElementById("notice").textContent = data.notice || "";
    results.hidden = false;
    status.textContent = "";
  } catch (error) {
    status.textContent = "Error: " + error.message;
  }
});
