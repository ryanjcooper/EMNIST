(function() {
	var canvas = document.querySelector("#canvas");
	var context = canvas.getContext("2d");
	canvas.width = 280;
	canvas.height = 280;

	var loc = {x:0, y:0};
	var prev = {x:0, y:0};
	context.fillStyle = "white";
	context.fillRect(0, 0, canvas.width, canvas.height);
	context.color = "black";
	context.lineWidth = 7;
    context.lineJoin = context.lineCap = 'round';

	debug();

	canvas.addEventListener("mousemove", function(e) {
		prev.x = loc.x;
		prev.y = loc.y;

		loc.x = e.pageX - this.offsetLeft-15;
		loc.y = e.pageY - this.offsetTop-15;
	}, false);
	canvas.addEventListener("mousedown", function(e) {
		canvas.addEventListener("mousemove", onPaint, false);
	}, false);
	canvas.addEventListener("mouseup", function() {
		canvas.removeEventListener("mousemove", onPaint, false);
	}, false);

	// Set up touch events for mobile, etc
	canvas.addEventListener("touchmove", function (e) {
	  var touch = e.touches[0];
	  var mouseEvent = new MouseEvent("mousemove", {
	    clientX: touch.clientX,
	    clientY: touch.clientY
	  });
	  canvas.dispatchEvent(mouseEvent);
	}, false);
	canvas.addEventListener("touchstart", function (e) {
	        loc = getTouchPos(canvas, e);
	  var touch = e.touches[0];
	  var mouseEvent = new MouseEvent("mousedown", {
	    clientX: touch.clientX,
	    clientY: touch.clientY
	  });
	  canvas.dispatchEvent(mouseEvent);
	}, false);
	canvas.addEventListener("touchend", function (e) {
	  var mouseEvent = new MouseEvent("mouseup", {});
	  canvas.dispatchEvent(mouseEvent);
	}, false);

	// Get the position of a touch relative to the canvas
	function getTouchPos(canvasDom, touchEvent) {
	  var rect = canvasDom.getBoundingClientRect();
	  return {
	    x: touchEvent.touches[0].clientX - rect.left,
	    y: touchEvent.touches[0].clientY - rect.top
	  };
	}

	// Prevent scrolling when touching the canvas
	document.body.addEventListener("touchstart", function (e) {
	  if (e.target == canvas) {
	    e.preventDefault();
	  }
	}, false);
	document.body.addEventListener("touchend", function (e) {
	  if (e.target == canvas) {
	    e.preventDefault();
	  }
	}, false);
	document.body.addEventListener("touchmove", function (e) {
	  if (e.target == canvas) {
	    e.preventDefault();
	  }
	}, false);

	var onPaint = function() {
		context.lineWidth = context.lineWidth;
		context.lineJoin = "round";
		context.lineCap = "round";
		context.strokeStyle = context.color;

		context.beginPath();
		context.moveTo(prev.x, prev.y);
		context.lineTo(loc.x,loc.y );
		context.closePath();
		context.stroke();
	};

	function debug() {
		$("#clear_button").on("click", function() {
			context.clearRect( 0, 0, 280, 280 );
			context.fillStyle="white";
			context.fillRect(0,0,canvas.width,canvas.height);
		});
	}
}());
