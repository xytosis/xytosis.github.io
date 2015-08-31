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


	$(".headerbutton").hover(function(){
		$(this).css("-webkit-filter", "invert(100%)");
		$(this).css("filter", "progid:DXImageTransform.Microsoft.BasicImage(invert='1')");
	}, 
	function(){
		$(this).css("-webkit-filter", "invert(0%)");
		$(this).css("filter", "progid:DXImageTransform.Microsoft.BasicImage(invert='0')");
	});

	$("#contactbutton").hover(function(){
		$(this).css("-webkit-filter", "invert(100%)");
		$(this).css("filter", "progid:DXImageTransform.Microsoft.BasicImage(invert='1')");
	}, 
	function(){
		$(this).css("-webkit-filter", "invert(0%)");
		$(this).css("filter", "progid:DXImageTransform.Microsoft.BasicImage(invert='0')");
	});

	$(".headerlink").hover(function(){
		$(this).css("opacity", "0.8");
	}, 
	function(){
		$(this).css("opacity", "1");
	});

});