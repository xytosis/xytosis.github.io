/* This script will contain formatting for main page*/

$(document).ready(function() {
	$(".headerbutton").css("height", $(".headerbutton").width()+"px");

	$("#headerbuttons").css("margin-top", ($("#specialheadertext").height() * 1.5) + "px");

	$("#specialheadertext").css("margin-top", (200 - $("#specialheadertext").height()) + "px");
});