var loader = function(e) {
    let file = e.target.files;
    let show = "<span>Selected file:</span>" + file[0].name;
    let output = document.getElementById("selector");
    output.innerHTML = show;
    output.classList.add("active");
};

let fileInput = document.getElementById("file");
fileInput.addEventListener("change", loader);

function reset() {
    let str = document.getElementById("file").value;
    let output = document.getElementById("selector");
    output.innerHTML = "<span>Select file</span>";
    output.classList.add("label");
    cosole.log(str);
}