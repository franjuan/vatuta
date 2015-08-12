/**
 * Vatuta library
 */

/**
 * Task
 * @constructor
 * @param {string} Description string for representation of task, no processing over it.
 * @param {number} Duration in days of task
 * @return A new task type
 */
function Task(description, duration) {
	this.description = description;
	this.duration = duration;
}
/**
 * Returns task's description
 * @returns {string}
 */
Task.prototype.getDescription = function() {
	return this.description;
}
/**
 * Returns task's duration in days
 * @returns {string}
 */
Task.prototype.getDuration = function() {
	return this.duration;
}

Task.prototype.getContainer = function() {
	var container = new createjs.Container();
	
	// Add a shape
	var shape = new createjs.Shape();
	shape.graphics.beginFill("DeepSkyBlue");
	shape.graphics.drawRect(0, 0, 10*this.getDuration(), 50);
	container.addChild(shape);
	
	// Add a label
	var text = new createjs.Text(this.getDescription(),"20px Arial", "White");
	text.textAlign="center";
	text.x=10*this.getDuration()/2;
	text.y=50/2;
	text.m
	container.addChild(text);

	return container;
}