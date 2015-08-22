/**
 * Vatuta library
 */

/**
 * Task
 * 
 * @constructor
 * @param {string}
 *            Description string for representation of task, no processing over
 *            it.
 * @param {number}
 *            Duration in days of task
 * @return A new task type
 */
function Task(description, duration) {
	this.description = description;
	this.duration = duration;
	this.container;
}
/**
 * Returns task's description
 * 
 * @returns {string}
 */
Task.prototype.getDescription = function() {
	return this.description;
}
/**
 * Returns task's duration in days
 * 
 * @returns {string}
 */
Task.prototype.getDuration = function() {
	return this.duration;
}

Task.prototype.getContainer = function() {
	if (!this.container) {
		this.container = new createjs.Container();

		// Add a shape
		var shape = new createjs.Shape();
		shape.graphics.beginFill("DeepSkyBlue");
		shape.graphics.drawRect(0, 0, 10 * this.getDuration(), 50);
		this.container.addChild(shape);

		// Add a label
		var text = new createjs.Text(this.getDescription(), "20px Arial",
				"White");
		text.textAlign = "center";
		text.x = 10 * this.getDuration() / 2;
		text.y = 50 / 2;
		text.m
		this.container.addChild(text);

		this.container.setBounds(0, 0, 10 * this.getDuration(), 50);
	}
	return this.container;
}

Task.prepareAfterCreating = function(task) {
	var oTask = new Task(task.description, task.duration);

	return oTask;
}

/**
 * Project
 * 
 * @constructor
 * @param {string}
 *            Name string of project, no processing over it.
 * @return A new project type
 */
function Project(name) {
	this.name = name;
};

Project.loadProjectFromJSON = function(url) {
	var project;
	$.getJSON(url, function(data) {
		project=data;
	});
	
	return Project.prepareAfterCreating(project);
}

Project.prepareAfterCreating = function(project) {
	var oProject = new Project(project.name);
	oProject.streams = [];
	
	project.streams.forEach(function(stream) {
		oProject.streams.push(Stream.prepareAfterCreating(stream));
	});

	return oProject;
}

/**
 * Stream
 * 
 * @constructor
 * @param {string}
 *            Name string of stream, no processing over it.
 * @return A new project type
 */
function Stream(name) {
	this.name = name;
}

Stream.prepareAfterCreating = function(stream) {
	var oStream = new Stream(stream.name)
	oStream.tasks = [];
	
	stream.tasks.forEach(function(task) {
		oStream.tasks.push(Task.prepareAfterCreating(task));
	});
	
	return oStream;
}