/* This script will contain formatting for main page*/

$(document).ready(function() {
	document.body.style.overflow = "hidden";
	document.body.style.overflow = "visible";
	$("#mobilemenu").click(function() {
		var headerbar = $("#mobilemenubar");
		if (headerbar.css("visibility") == "visible") {
			headerbar.css("visibility", "hidden");
			document.body.style.overflow = "visible";
		} else {
			headerbar.css("visibility", "visible");
			document.body.style.overflow = "hidden";
		}
		
	})

});